#!/usr/bin/env python

import collections
import pyaudio
import snowboydetect

import wave
import os
import logging
from ctypes import *
from contextlib import contextmanager


from utils.general import load_dataset, normalize_img, post_process, compare_embadding, cv2ImgAddText
from utils.extract_face import extract

import torch
import cv2
import numpy as np
import onnxruntime
import time



logging.basicConfig()
logger = logging.getLogger("snowboy")
logger.setLevel(logging.INFO)
TOP_DIR = os.path.dirname(os.path.abspath(__file__))

RESOURCE_FILE = os.path.join(TOP_DIR, "resources/common.res")
DETECT_DING = os.path.join(TOP_DIR, "resources/ding.wav")
DETECT_DONG = os.path.join(TOP_DIR, "resources/dong.wav")

def py_error_handler(filename, line, function, err, fmt):
    pass

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def no_alsa_error():
    try:
        asound = cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(c_error_handler)
        yield
        asound.snd_lib_error_set_handler(None)
    except:
        yield
        pass

class RingBuffer(object):
    """Ring buffer to hold audio from PortAudio"""

    def __init__(self, size=4096):
        self._buf = collections.deque(maxlen=size)

    def extend(self, data):
        """Adds data to the end of buffer"""
        self._buf.extend(data)

    def get(self):
        """Retrieves data from the beginning of buffer and clears it"""
        tmp = bytes(bytearray(self._buf))
        self._buf.clear()
        return tmp


def play_audio_file(fname=DETECT_DING):
    """Simple callback function to play a wave file. By default it plays
    a Ding sound.

    :param str fname: wave file name
    :return: None
    """
    ding_wav = wave.open(fname, 'rb')
    ding_data = ding_wav.readframes(ding_wav.getnframes())
    with no_alsa_error():
        audio = pyaudio.PyAudio()
    stream_out = audio.open(
        format=audio.get_format_from_width(ding_wav.getsampwidth()),
        channels=ding_wav.getnchannels(),
        rate=ding_wav.getframerate(), input=False, output=True)
    stream_out.start_stream()
    stream_out.write(ding_data)
    time.sleep(0.2)
    stream_out.stop_stream()
    stream_out.close()
    audio.terminate()


class HotwordDetector(object):
    """
    Snowboy decoder to detect whether a keyword specified by `decoder_model`
    exists in a microphone input stream.

    :param decoder_model: decoder model file path, a string or a list of strings
    :param resource: resource file path.
    :param sensitivity: decoder sensitivity, a float of a list of floats.
                              The bigger the value, the more senstive the
                              decoder. If an empty list is provided, then the
                              default sensitivity in the model will be used.
    :param audio_gain: multiply input volume by this factor.
    :param apply_frontend: applies the frontend processing algorithm if True.
    """

    def __init__(self, decoder_model,
                 resource=RESOURCE_FILE,
                 sensitivity=[],
                 audio_gain=1,
                 apply_frontend=False):

        tm = type(decoder_model)
        ts = type(sensitivity)
        if tm is not list:
            decoder_model = [decoder_model]
        if ts is not list:
            sensitivity = [sensitivity]
        model_str = ",".join(decoder_model)

        self.detector = snowboydetect.SnowboyDetect(
            resource_filename=resource.encode(), model_str=model_str.encode())
        self.detector.SetAudioGain(audio_gain)
        self.detector.ApplyFrontend(apply_frontend)
        self.num_hotwords = self.detector.NumHotwords()

        if len(decoder_model) > 1 and len(sensitivity) == 1:
            sensitivity = sensitivity * self.num_hotwords
        if len(sensitivity) != 0:
            assert self.num_hotwords == len(sensitivity), \
                "number of hotwords in decoder_model (%d) and sensitivity " \
                "(%d) does not match" % (self.num_hotwords, len(sensitivity))
        sensitivity_str = ",".join([str(t) for t in sensitivity])
        if len(sensitivity) != 0:
            self.detector.SetSensitivity(sensitivity_str.encode())

        self.ring_buffer = RingBuffer(
            self.detector.NumChannels() * self.detector.SampleRate() * 5)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(onnxruntime.get_device()))

        dataset_path = 'data/emb/faceEmbedding.npy'
        filename = 'data/emb/name.txt'
        self.dataset_embeddings, self.names_list = load_dataset(dataset_path, filename)

        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        self.face_detect_session = onnxruntime.InferenceSession("checkpoints/yolov7-lite-s.onnx", None, providers=providers)
        self.face_recognition_session = onnxruntime.InferenceSession("checkpoints/20180402-114759-vggface2.onnx", None,
                                                                providers=providers)

        self.cap = cv2.VideoCapture(1)

        self.start_time = time.time()
        self.counter = 0

    def start(self, detected_callback=play_audio_file,
              interrupt_check=lambda: False,
              sleep_time=0.03,
              audio_recorder_callback=None,
              silent_count_threshold=15,
              recording_timeout=100):
        """
        Start the voice detector. For every `sleep_time` second it checks the
        audio buffer for triggering keywords. If detected, then call
        corresponding function in `detected_callback`, which can be a single
        function (single model) or a list of callback functions (multiple
        models). Every loop it also calls `interrupt_check` -- if it returns
        True, then breaks from the loop and return.

        :param detected_callback: a function or list of functions. The number of
                                  items must match the number of models in
                                  `decoder_model`.
        :param interrupt_check: a function that returns True if the main loop
                                needs to stop.
        :param float sleep_time: how much time in second every loop waits.
        :param audio_recorder_callback: if specified, this will be called after
                                        a keyword has been spoken and after the
                                        phrase immediately after the keyword has
                                        been recorded. The function will be
                                        passed the name of the file where the
                                        phrase was recorded.
        :param silent_count_threshold: indicates how long silence must be heard
                                       to mark the end of a phrase that is
                                       being recorded.
        :param recording_timeout: limits the maximum length of a recording.
        :return: None
        """
        self._running = True

        def audio_callback(in_data, frame_count, time_info, status):
            self.ring_buffer.extend(in_data)
            play_data = chr(0) * len(in_data)
            return play_data, pyaudio.paContinue

        with no_alsa_error():
            self.audio = pyaudio.PyAudio()
        self.stream_in = self.audio.open(
            input=True, output=False,
            format=self.audio.get_format_from_width(
                self.detector.BitsPerSample() / 8),
            channels=self.detector.NumChannels(),
            rate=self.detector.SampleRate(),
            frames_per_buffer=2048,
            stream_callback=audio_callback)

        if interrupt_check():
            logger.debug("detect voice return")
            return

        tc = type(detected_callback)
        if tc is not list:
            detected_callback = [detected_callback]
        if len(detected_callback) == 1 and self.num_hotwords > 1:
            detected_callback *= self.num_hotwords

        assert self.num_hotwords == len(detected_callback), \
            "Error: hotwords in your models (%d) do not match the number of " \
            "callbacks (%d)" % (self.num_hotwords, len(detected_callback))

        logger.debug("detecting...")

        state = "PASSIVE"
        while self._running is True:

            _, im = self.cap.read()
            im_bgr = im.copy()
            x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            x_onnx = normalize_img(x, 0.0, 0.00392156862745098)
            detect_input_name = self.face_detect_session.get_inputs()[0].name
            detect_output = self.face_detect_session.run([], {detect_input_name: x_onnx})[0]
            batch_boxes, prob, _ = post_process(torch.from_numpy(detect_output).to(self.device), x_onnx, im)

            # batch_boxes, prob = yolov7_face(x, return_prob=True)
            self.counter += 1
            if len(batch_boxes) != 0:
                # print('Face detected with probability: {:8f}'.format(prob[0]))
                try:
                    x_aligned = extract(x, batch_boxes, keep_all=True)
                    # x_aligned = x_aligned.unsqueeze(0)
                    pred_embeddings = []
                    for i in range(len(x_aligned)):
                        recog_input_name = self.face_recognition_session.get_inputs()[0].name
                        pred_embeddings.append(
                            self.face_recognition_session.run([], {recog_input_name: np.array(x_aligned[i].unsqueeze(0))})[
                                0])
                    pred_name, pred_score = compare_embadding(np.array(pred_embeddings), self.dataset_embeddings, self.names_list)
                    show_info = [n + ':' + str(s)[:5] for n, s in zip(pred_name, pred_score)]

                    for name, box in zip(show_info, batch_boxes):
                        box = [int(b) for b in box]
                        cv2.rectangle(im_bgr, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2, 8, 0)
                        im_bgr = cv2ImgAddText(im_bgr, name, box[0], box[1], textColor=(255,0,0), textSize=30)
                        # cv2.putText(im_bgr, name, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                        #             thickness=2)
                except:
                    continue
            fps = self.counter / (time.time() - self.start_time)
            fpsString = "FPS: {0}".format(float("%.2f" % fps))
            cv2.putText(im_bgr, fpsString, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.imshow("face_recognition", im_bgr)
            keycode = cv2.waitKey(1) & 0xFF
            if keycode == 27:
                break
            self.counter = 0
            self.start_time = time.time()


            if interrupt_check():
                logger.debug("detect voice break")
                break
            data = self.ring_buffer.get()
            if len(data) == 0:
                time.sleep(sleep_time)
                continue

            status = self.detector.RunDetection(data)
            if status == -1:
                logger.warning("Error initializing streams or reading audio data")

            #small state machine to handle recording of phrase after keyword
            if state == "PASSIVE":
                if status > 0: #key word found
                    self.recordedData = []
                    self.recordedData.append(data)
                    silentCount = 0
                    recordingCount = 0
                    message = "Keyword " + str(status) + " detected at time: "
                    message += time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(time.time()))
                    logger.info(message)
                    callback = detected_callback[status-1]
                    if callback is not None:
                        callback()

                    if audio_recorder_callback is not None:
                        state = "ACTIVE"
                    continue

            elif state == "ACTIVE":
                stopRecording = False
                if recordingCount > recording_timeout:
                    stopRecording = True
                elif status == -2: #silence found
                    if silentCount > silent_count_threshold:
                        stopRecording = True
                    else:
                        silentCount = silentCount + 1
                elif status == 0: #voice found
                    silentCount = 0

                if stopRecording == True:

                    if len(batch_boxes) != 0:
                        text = ""
                        for i in range(len(pred_name)):
                            if pred_name[i] != "unknown":
                                text += "你好" + pred_name[i] + ","
                        audio_recorder_callback(text)
                    state = "PASSIVE"
                    continue

                recordingCount = recordingCount + 1
                self.recordedData.append(data)

        logger.debug("finished.")

    def saveMessage(self):
        """
        Save the message stored in self.recordedData to a timestamped file.
        """
        filename = 'output' + str(int(time.time())) + '.wav'
        data = b''.join(self.recordedData)

        #use wave to save data
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(
            self.audio.get_format_from_width(
                self.detector.BitsPerSample() / 8)))
        wf.setframerate(self.detector.SampleRate())
        wf.writeframes(data)
        wf.close()
        logger.debug("finished saving: " + filename)
        return filename

    def terminate(self):
        """
        Terminate audio stream. Users can call start() again to detect.
        :return: None
        """
        self.stream_in.stop_stream()
        self.stream_in.close()
        self.audio.terminate()
        self._running = False
