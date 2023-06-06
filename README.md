# FaceRecognitionForJetsonNano
这是一个用于Nvidia Jetson系列的人脸识别算法，能够通过语音唤醒词“小绿小绿”使机器说出当前看到人的名字
## 依赖
通过以下命令添加语音库<br>
    sudo apt update<br>
    sudo apt install python3-pip<br>
    sudo apt-get install sox libsox-fmt-all<br>
    sudo apt-get install portaudio19-dev<br>
    sudo apt-get install libatlas-base-dev<br>
    sudo apt-get install flac<br>
## 更新唤醒词
安装虚拟机，刻录该链接的镜像[baidu](https://pan.baidu.com/s/17WGj_gtp9xAT4-UEw1rQxQ)，提取码：k46u<br>
<br>
使用以下命令来训练你的声音模型<br>
cd snowboy/examples/Python<br>
rec -r 16000 -c 1 -b 16 -e signed-integer -t wav record1.wav # 录音三次<br>
python generate_pmdl.py -r1=record1.wav -r2=record2.wav -r3=record3.wav -lang=en -n=hotword.pmdl<br>
<br>
把本代码中的hotword.pmdl文件替换为你训练好的hotword.pmdl文件<br>
## 添加科大讯飞APIkey
在[科大讯飞开放平台](https://www.xfyun.cn/)中找到APPID，APISecret，APIKey<br>
在文件text2audio.py中141行处添加APPID，APISecret，APIKey
## 录入人脸识别信息
### 采集照片
python collect_image.py
### 创建数据库
python create_dataset.py
## 运行示例
python demo.py
