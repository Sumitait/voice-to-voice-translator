
import os
import uuid
from pathlib import Path
import numpy as np
import gradio as gr
import assemblyai as aai
from translate import Translator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

def transcribe_audio(audio_file):
    aai.settings.api_key = "API Key "                  // add assemblyai API key
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)
    return transcript


def translate_text(text: str) -> list:
    languages = ["ru", "tr", "sv", "de", "es", "ja"]
   # languages = [ "sv",  "ja"]
    list_translations = []
    for lan in languages:
        translator = Translator(from_lang="en", to_lang=lan)
        translation = translator.translate(text)
        list_translations.append(translation)
    return list_translations

def text_to_speech(text: str) -> str:
    client = ElevenLabs(api_key="API key")                                         //add ElevenLads API key
    response = client.text_to_speech.convert(
        # voice_id="VOice ID API key",                                            // add voice id for ElevenLAbs
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.8),
    )
    save_file_path = f"{uuid.uuid4()}.mp3"
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
    print(f"Audio file saved: {save_file_path}")
    return save_file_path

def voice_to_voice(audio_file):
    transcript = transcribe_audio(audio_file)
    if transcript.status == aai.TranscriptStatus.error:
        raise gr.Error(transcript.error)
    transcript_text = transcript.text
    translations = translate_text(transcript_text)
    audio_paths = [text_to_speech(t) for t in translations]
    return (*audio_paths, *translations)
with gr.Blocks() as demo:
    gr.Markdown("## Record in English and receive translations!")
    with gr.Row():

        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Audio")
        submit = gr.Button("Submit")
        btn = gr.Button("Clear")
        btn.click(lambda: None, None, audio_input)

    with gr.Row():
        outputs = [
           gr.Audio(label="Russian"),
            gr.Audio(label="Turkish"),
            gr.Audio(label="Swedish"),
            gr.Audio(label="German"),
            gr.Audio(label="Spanish"),
            gr.Audio(label="Japanese"),
        ]
        texts = [gr.Textbox(label="Text Output") for _ in outputs]
    
    submit.click(fn=voice_to_voice, inputs=audio_input, outputs=outputs + texts)

if __name__ == "__main__":
    demo.launch()
