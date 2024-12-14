import os
import uuid
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic_settings import BaseSettings
import google.generativeai as genai
import mimetypes
import tempfile
import shutil

class Settings(BaseSettings):
    gemini_api_key: str
    temp_dir: str = "temp"
    
    class Config:
        env_file = ".env"

class MediaProcessor:
    SUPPORTED_VIDEO_FORMATS = {
        'video/mp4', 'video/mpeg', 'video/quicktime', 
        'video/x-msvideo', 'video/x-matroska'
    }
    
    SUPPORTED_AUDIO_FORMATS = {
        'audio/wav', 'audio/mp3', 'audio/aiff', 
        'audio/aac', 'audio/ogg', 'audio/flac'
    }
    
    def __init__(self, temp_dir: str):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
    def _check_ffmpeg(self):
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True)
            return True
        except FileNotFoundError:
            return False
    
    def cleanup_temp_files(self, *files):
        for file in files:
            try:
                if file and os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                print(f"Error cleaning up {file}: {e}")
    
    async def process_media(self, file: UploadFile) -> tuple[bytes, str]:
        """处理上传的媒体文件，返回音频内容和MIME类型"""
        if not file.content_type:
            mime_type, _ = mimetypes.guess_type(file.filename)
            if not mime_type:
                raise ValueError("无法确定文件类型")
        else:
            mime_type = file.content_type
            
        # 为临时文件生成唯一文件名
        temp_input_path = self.temp_dir / f"{uuid.uuid4()}{Path(file.filename).suffix}"
        temp_output_path = self.temp_dir / f"{uuid.uuid4()}.mp3"
        
        try:
            # 保存上传的文件
            content = await file.read()
            with open(temp_input_path, "wb") as f:
                f.write(content)
            
            # 如果是视频文件，转换为音频
            if mime_type in self.SUPPORTED_VIDEO_FORMATS:
                if not self._check_ffmpeg():
                    raise ValueError("未安装ffmpeg，无法处理视频文件")
                    
                # 使用ffmpeg提取音频
                command = [
                    'ffmpeg', '-i', str(temp_input_path),
                    '-vn',  # 禁用视频
                    '-acodec', 'libmp3lame',  # 使用MP3编码器
                    '-ar', '44100',  # 采样率
                    '-ac', '2',  # 声道数
                    '-b:a', '192k',  # 比特率
                    str(temp_output_path)
                ]
                
                process = subprocess.run(command, capture_output=True)
                if process.returncode != 0:
                    raise ValueError(f"视频处理失败: {process.stderr.decode()}")
                
                # 读取处理后的音频文件
                with open(temp_output_path, 'rb') as f:
                    audio_content = f.read()
                return audio_content, 'audio/mp3'
                
            # 如果是音频文件，直接返回
            elif mime_type in self.SUPPORTED_AUDIO_FORMATS:
                return content, mime_type
            else:
                raise ValueError(f"不支持的文件格式: {mime_type}")
                
        finally:
            # 清理临时文件
            self.cleanup_temp_files(temp_input_path, temp_output_path)

class GeminiASR:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    async def transcribe_audio(self, file_content: bytes, mime_type: str) -> str:
        try:
            prompt = "Generate a verbatim transcript of the audio. Output only the transcribed text without any additional explanations or formatting."
            
            audio_data = {
                "mime_type": mime_type,
                "data": file_content
            }
            
            response = self.model.generate_content([prompt, audio_data])
            return response.text.strip()
            
        except Exception as e:
            raise Exception(f"转写错误: {str(e)}")

app = FastAPI(
    title="音视频转文字 API",
    description="基于 Gemini 的音视频转写服务",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

try:
    settings = Settings()
except Exception:
    raise Exception("未找到 GEMINI_API_KEY 环境变量。请确保已经设置了环境变量或创建了.env文件。")

media_processor = MediaProcessor(settings.temp_dir)
asr = GeminiASR(settings.gemini_api_key)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>音视频转文字 API 文档</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }
            .endpoint {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }
            .method {
                background: #4CAF50;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 14px;
            }
            code {
                background: #e9ecef;
                padding: 2px 6px;
                border-radius: 4px;
                font-family: monospace;
            }
            .note {
                border-left: 4px solid #ffc107;
                padding-left: 15px;
                margin: 15px 0;
                color: #666;
            }
        </style>
    </head>
    <body>
        <h1>🎙️ 音视频转文字 API 文档</h1>
        
        <p>这是一个基于 Gemini 的音视频转写服务，支持多种格式的音视频转写功能。</p>
        
        <div class="endpoint">
            <span class="method">POST</span>
            <code>/api/v1/transcribe</code>
            
            <h3>接口说明</h3>
            <p>将音视频文件转换为文字。</p>
            
            <h3>请求格式</h3>
            <ul>
                <li>请求方式：POST</li>
                <li>Content-Type: multipart/form-data</li>
                <li>参数名：file（音视频文件）</li>
            </ul>
            
            <h3>支持的视频格式</h3>
            <ul>
                <li>MP4 (.mp4)</li>
                <li>MPEG (.mpeg, .mpg)</li>
                <li>QuickTime (.mov)</li>
                <li>AVI (.avi)</li>
                <li>Matroska (.mkv)</li>
            </ul>
            
            <h3>支持的音频格式</h3>
            <ul>
                <li>WAV (.wav)</li>
                <li>MP3 (.mp3)</li>
                <li>AAC (.aac)</li>
                <li>OGG (.ogg)</li>
                <li>FLAC (.flac)</li>
                <li>AIFF (.aiff)</li>
            </ul>
            
            <h3>限制条件</h3>
            <div class="note">
                <p>• 文件大小上限：20MB</p>
                <p>• 需要安装ffmpeg才能处理视频文件</p>
                <p>• 仅支持上述列出的音视频格式</p>
            </div>
            
            <h3>返回格式</h3>
            <pre><code>{
    "transcript": "转写的文本内容"
}</code></pre>
            
            <h3>错误码</h3>
            <ul>
                <li>400：请求参数错误（文件格式不支持、文件过大等）</li>
                <li>500：服务器内部错误</li>
            </ul>
        </div>
        
        <footer style="text-align: center; margin-top: 40px; color: #666;">
            <p>© 2024 音视频转文字 API 服务</p>
        </footer>
    </body>
    </html>
    """

@app.post("/api/v1/transcribe")
async def transcribe_media(file: UploadFile):
    """
    转写音视频文件为文本
    """
    try:
        file_size = 0
        content = await file.read()
        file_size = len(content)
        await file.seek(0)  # 重置文件指针
        
        if file_size > 20 * 1024 * 1024:  # 20MB
            raise HTTPException(status_code=400, detail="文件大小超过20MB限制")
        
        try:
            audio_content, mime_type = await media_processor.process_media(file)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
        transcript = await asr.transcribe_audio(audio_content, mime_type)
        return {"transcript": transcript}
        
    except Exception as e:
        if "未安装ffmpeg" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=2323)