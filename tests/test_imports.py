try:
    import torch
    print("torch ok")
    import transformers
    print("transformers ok")
    import peft
    print("peft ok")
    import langchain_community
    print("langchain_community ok")
    import langchain_huggingface
    print("langchain_huggingface ok")
    import langchain_ollama
    print("langchain_ollama ok")
    import cv2
    print("cv2 ok")
    import librosa
    print("librosa ok")
    import main
    print("main ok")
except Exception as e:
    import traceback
    traceback.print_exc()
