"""
Offline AI Notes Summarizer
Transcribes and summarizes video/audio without internet
"""

import os
from faster_whisper import WhisperModel
from transformers import pipeline
from moviepy.editor import VideoFileClip
import warnings
warnings.filterwarnings('ignore')

class OfflineNotesSummarizer:
    def __init__(self, whisper_model_size="small", device="cpu"):
        """
        Initialize the summarizer
        whisper_model_size: tiny, base, small, medium, large
        device: cpu or cuda
        """
        print(f"🔧 Loading Whisper model ({whisper_model_size})...")
        self.whisper_model = WhisperModel(whisper_model_size, device=device, compute_type="int8")
        
        print("🔧 Loading summarization model...")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        print("✅ Models loaded! Ready to process.")

    def extract_audio(self, video_path, audio_path="temp_audio.mp3"):
        """Extract audio from video file"""
        print(f"🎵 Extracting audio from {video_path}...")
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, logger=None)
            video.close()
            print("✅ Audio extracted!")
            return audio_path
        except Exception as e:
            print(f"❌ Error extracting audio: {e}")
            return None

    def transcribe(self, audio_path, language="en"):
        """Transcribe audio to text"""
        print(f"📝 Transcribing audio (language: {language})...")
        segments, info = self.whisper_model.transcribe(
            audio_path,
            language=language,
            beam_size=5
        )
        
        transcript = ""
        timestamps = []
        
        for segment in segments:
            transcript += segment.text + " "
            timestamps.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
        
        print(f"✅ Transcription complete! Detected language: {info.language}")
        return transcript.strip(), timestamps

    def summarize(self, text, max_length=130):
        """Summarize text into key points only"""
        print("🤖 Generating key points summary...")
        
        # Split long text into chunks
        max_chunk_length = 1024
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            if current_length >= max_chunk_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Summarize each chunk - extract key points only
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  Extracting key points from chunk {i+1}/{len(chunks)}...")
            if len(chunk.split()) > 50:
                summary = self.summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=25,
                    do_sample=False,
                    truncation=True
                )
                summaries.append(summary[0]['summary_text'])
        
        print("✅ Key points extracted!")
        # Format as bullet points
        formatted_summary = self.format_bullet_points(summaries)
        return formatted_summary

    def format_bullet_points(self, summaries):
        """Format summaries as bullet points"""
        bullet_points = []
        for summary in summaries:
            # Split into sentences
            sentences = summary.replace('. ', '.|').split('|')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 15: # Only meaningful sentences
                    bullet_points.append(f"• {sentence}")
        return "\n".join(bullet_points)

    def process_video(self, video_path, language="en", output_file="notes.txt"):
        """Complete pipeline: video → audio → transcript → summary"""
        print(f"\n{'='*60}")
        print(f"🎬 Processing: {video_path}")
        print(f"{'='*60}\n")
        
        # Extract audio
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return None
        
        # Transcribe
        transcript, timestamps = self.transcribe(audio_path, language)
        
        # Summarize
        summary = self.summarize(transcript)
        
        # Save results
        self.save_notes(transcript, summary, timestamps, output_file)
        
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return {"transcript": transcript, "summary": summary, "timestamps": timestamps}

    def save_notes(self, transcript, summary, timestamps, output_file):
        """Save formatted notes to file"""
        print(f"💾 Saving notes to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("📝 IMPORTANT POINTS - AI NOTES SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("🎯 KEY TAKEAWAYS\n")
            f.write("-" * 60 + "\n\n")
            f.write(summary + "\n\n")
            f.write("\n" + "=" * 60 + "\n\n")
            
            f.write("💡 NOTE: Full transcript available if needed\n")
            f.write("The above points capture the essential information.\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("⏱️ TIMESTAMPS\n")
            f.write("=" * 60 + "\n\n")
            
            # Show only key timestamps (every 5th one for overview)
            for i, ts in enumerate(timestamps):
                if i % 5 == 0 and i < 40: # Show fewer, more important points
                    minutes = int(ts["start"] // 60)
                    seconds = int(ts["start"] % 60)
                    f.write(f"[{minutes:02d}:{seconds:02d}] {ts['text']}\n")
        
        print(f"✅ Notes saved successfully!")

# Usage Example
if __name__ == "__main__":
    # Initialize summarizer
    summarizer = OfflineNotesSummarizer(
        whisper_model_size="small", # tiny/base/small/medium/large
        device="cpu" # or "cuda" for GPU
    )
    
    # Process a video - USE FULL PATH!
    video_path = r"C:\Users\roopa\test123\lecture.mp4" # CHANGE THIS!
    
    # Check if file exists
    import os
    if not os.path.exists(video_path):
        print(f"❌ ERROR: Video file not found at: {video_path}")
        print("\n💡 HOW TO FIX:")
        print("1. Right-click your video file in File Explorer")
        print("2. Click 'Copy as path'")
        print("3. Paste it in the code above (line with video_path)")
        print("4. Keep the r before the quotes: r\"your_path_here\"")
        exit()
    
    results = summarizer.process_video(
        video_path,
        language="en", # en/hi/kn for English/Hindi/Kannada
        output_file="lecture_notes.txt"
    )
    
    if results:
        print(f"\n{'='*60}")
        print("🎉 Processing Complete!")
        print(f"{'='*60}\n")
        print("🎯 KEY POINTS EXTRACTED:")
        print("-" * 60)
        print(results["summary"])
        print("\n✅ Important points saved to lecture_notes.txt")
    else:
        print("❌ Processing failed. Check the error messages above.")