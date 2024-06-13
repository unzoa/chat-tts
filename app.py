import ChatTTS
# from pydub import AudioSegment
import torch
import torchaudio

chat = ChatTTS.Chat()
chat.load_models()

text = 'hello'

# 文本转为音频
wavs = chat.infer(text)

# 保存音频文件到本地文件（采样率为24000Hz）
torchaudio.save("./output/output-01.wav", torch.from_numpy(wavs[0]), 24000)

# texts = ["你好，世界！", "今天天气真不错！", "我很高兴见到你！"]
# for i, text in enumerate(texts):
#     audio = chat.synthesize(text)

#     # 将音频保存为 MP3 格式
#     output_file = f"output_{i}.mp3"
#     audio.export(output_file, format="mp3")

#     print(f"Audio saved to {output_file}")
