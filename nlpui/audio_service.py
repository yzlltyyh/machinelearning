import sounddevice as sd
import numpy as np
import threading
import queue
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from flask import Flask, jsonify
from flask_socketio import SocketIO

# 配置DashScope
dashscope.api_key = 'sk-0c094c3ea1374da8b150fbe6753d4067'

class AudioRecorder:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # 列出可用的音频设备
        print("Available audio devices:")
        print(sd.query_devices())
        
        # 尝试找到默认输入设备
        try:
            device_info = sd.query_devices(kind='input')
            print(f"Using input device: {device_info}")
        except Exception as e:
            print(f"Warning: No input device found - {e}")
        
    def start_recording(self):
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()
        
    def stop_recording(self):
        self.is_recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()
        
    def _record(self):
        try:
            def callback(indata, frames, time, status):
                if status:
                    print(f'Recording error: {status}')
                if self.is_recording:
                    # 将数据转换为正确的格式
                    audio_data = (indata * 32767).astype(np.int16).tobytes()
                    self.audio_queue.put(audio_data)

            with sd.InputStream(samplerate=self.sample_rate,
                              channels=1,
                              dtype=np.float32,
                              callback=callback,
                              blocksize=self.chunk_size):
                while self.is_recording:
                    sd.sleep(100)  # 防止CPU过载
                    
        except Exception as e:
            print(f"Recording error: {e}")
            self.is_recording = False

class ASRCallback(RecognitionCallback):
    def __init__(self, socketio):
        self.socketio = socketio
        
    def on_event(self, result: RecognitionResult) -> None:
        try:
            if hasattr(result, 'get_text') and result.get_text():
                self.socketio.emit('recognition_result', {
                    'text': result.get_text(),
                    'is_final': result.is_final
                })
        except Exception as e:
            print(f"Error in on_event: {e}")
    
    def on_error(self, result: RecognitionResult) -> None:
        try:
            error_msg = str(result.output.message if hasattr(result, 'output') else "Unknown error")
            self.socketio.emit('recognition_error', {
                'error': error_msg
            })
        except Exception as e:
            print(f"Error in on_error: {e}")
        
    def on_complete(self) -> None:
        try:
            self.socketio.emit('recognition_complete')
        except Exception as e:
            print(f"Error in on_complete: {e}")

class AudioService:
    def __init__(self, socketio):
        self.socketio = socketio
        self.recorder = AudioRecorder()
        self.callback = ASRCallback(socketio)
        self.recognition = Recognition(
            model='paraformer-realtime-v2',
            format='pcm',
            sample_rate=16000,
            callback=self.callback
        )
        print("AudioService initialized")
        
    def start_recognition(self):
        try:
            print("Starting recognition...")
            self.recorder.start_recording()
            self.recognition.start()
            
            # 创建一个新线程来处理音频数据
            self.process_thread = threading.Thread(target=self._process_audio)
            self.process_thread.start()
            print("Recognition started successfully")
            
        except Exception as e:
            print(f"Error starting recognition: {str(e)}")
            self.socketio.emit('recognition_error', {'error': str(e)})
    
    def _process_audio(self):
        try:
            print("Audio processing started")
            while self.recorder.is_recording:
                try:
                    audio_data = self.recorder.audio_queue.get(timeout=1)
                    self.recognition.send_audio(audio_data)
                except queue.Empty:
                    continue
            print("Audio processing stopped")
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            self.socketio.emit('recognition_error', {'error': str(e)})
                
    def stop_recognition(self):
        try:
            print("Stopping recognition...")
            self.recorder.stop_recording()
            if hasattr(self, 'process_thread'):
                self.process_thread.join()
            self.recognition.stop()
            print("Recognition stopped successfully")
        except Exception as e:
            print(f"Error stopping recognition: {str(e)}")
            self.socketio.emit('recognition_error', {'error': str(e)}) 