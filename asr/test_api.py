import requests
import os
from pathlib import Path
import glob

def test_asr_api(file_path: str, api_url: str = "http://127.0.0.1:2323"):
    """测试音视频转文字API"""
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误：文件 {file_path} 不存在")
        return

    # 准备文件
    files = {
        'file': open(file_path, 'rb')
    }
    
    try:
        print(f"\n🎯 正在测试文件：{Path(file_path).name}")
        print("----------------------------------")
        
        # 发送请求
        response = requests.post(
            f"{api_url}/api/v1/transcribe",
            files=files
        )
        
        # 检查响应
        if response.status_code == 200:
            result = response.json()
            print("✅ 测试成功！")
            
            # 显示原始转写结果
            print("\n📝 原始转写结果：")
            print("----------------------------------")
            print(result['transcript'])
            print("----------------------------------")
            
            # 显示断句结果
            if 'segments' in result and result['segments']:
                print("\n📋 断句结果：")
                print("----------------------------------")
                for i, segment in enumerate(result['segments'], 1):
                    print(f"{i}. {segment}")
                print("----------------------------------")
                print(f"共 {len(result['segments'])} 个片段")
            else:
                print("\n⚠️ 未返回断句结果")
                
        else:
            print(f"❌ 请求失败 (状态码: {response.status_code})")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 连接错误：无法连接到服务器，请确保服务器正在运行")
    except Exception as e:
        print(f"❌ 测试过程中出现错误：{str(e)}")
    finally:
        files['file'].close()

def find_media_files(directory: str) -> tuple[list, list]:
    """查找目录中的音视频文件"""
    audio_extensions = ['.wav', '.mp3', '.aac', '.ogg', '.flac', '.aiff']
    video_extensions = ['.mp4', '.mpeg', '.mpg', '.mov', '.avi', '.mkv']
    
    audio_files = []
    video_files = []
    
    # 查找音频文件
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        
    # 查找视频文件
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
    
    return audio_files, video_files

def main():
    """主测试函数"""
    print("🎙️ 音视频转文字 API 测试工具")
    print("==================================")
    
    # 创建测试文件目录
    test_dir = "test_files"
    os.makedirs(test_dir, exist_ok=True)
    
    # 查找所有音视频文件
    audio_files, video_files = find_media_files(test_dir)
    all_files = audio_files + video_files
    
    if not all_files:
        print("⚠️ 警告：未找到测试文件")
        print(f"请在 {test_dir} 目录下放置测试文件")
        
        print("\n支持的视频格式：")
        print("- MP4 (.mp4)")
        print("- MPEG (.mpeg, .mpg)")
        print("- QuickTime (.mov)")
        print("- AVI (.avi)")
        print("- Matroska (.mkv)")
        
        print("\n支持的音频格式：")
        print("- WAV (.wav)")
        print("- MP3 (.mp3)")
        print("- AAC (.aac)")
        print("- OGG (.ogg)")
        print("- FLAC (.flac)")
        print("- AIFF (.aiff)")
        
        # 显示当前目录中的所有文件，帮助调试
        print("\n当前目录中的文件：")
        all_dir_files = os.listdir(test_dir)
        if all_dir_files:
            for file in all_dir_files:
                print(f"- {file}")
        else:
            print("(目录为空)")
        return
    
    # 执行测试
    if audio_files:
        print(f"\n找到 {len(audio_files)} 个音频文件:")
        for file in audio_files:
            print(f"- {os.path.basename(file)}")
            
    if video_files:
        print(f"\n找到 {len(video_files)} 个视频文件:")
        for file in video_files:
            print(f"- {os.path.basename(file)}")
    
    print("\n开始测试...")
    for media_file in all_files:
        test_asr_api(media_file)

if __name__ == "__main__":
    main() 