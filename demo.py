import snowboydecoder
import text2audio

interrupted = False

def text2AudioCallback(text):
    print("converting text to audio")
    try:
        text2audio.main(text)
    except:
        print("convert failed")
    print("listening")

def detectedCallback():
    snowboydecoder.play_audio_file()
    print("recording audio...", end="", flush=True)

def signal_handler():
    global interrupted
    interrupted = True

def interrupt_callback():
    global interrupted
    return interrupted

model = "hotword.pmdl"

detecter = snowboydecoder.HotwordDetector(model, sensitivity=0.5)
print("Listening... Press Ctrl+C to exit")

detecter.start(detected_callback=detectedCallback,
               audio_recorder_callback=text2AudioCallback,
               interrupt_check=interrupt_callback,
               silent_count_threshold=0.1,
               sleep_time=0.03)

detecter.terminate()