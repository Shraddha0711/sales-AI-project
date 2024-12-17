import os
import re
import sys
import json
import base64
import wave
import argparse
import asyncio
from typing import Dict
from urllib.parse import urlparse
import firebase_admin
from firebase_admin import firestore, credentials
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport, DailyTranscriptionSettings

# Load environment variables
load_dotenv(override=True)

# Setup logger
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Firebase initialization
FILES_DIR = "saved_files"
cred = credentials.Certificate(os.getenv("CRED_PATH"))
firebase_admin.initialize_app(cred)
db = firestore.client()

# Save transcription data to Firebase
async def save_in_db(room_id: str, transcript: str):
    doc_ref = db.collection("Transcription").document(room_id)
    prompt_type = next((msg['content'].split(':')[-1].strip() for msg in transcript if msg.get('role') == 'system' and 'prompt_type' in msg['content']), None)
    transcription = re.sub(r'\.?\s*prompt_type:[a-zA-Z]+$', '', transcript[0]['content'])
    data = {"transcript": transcription, "type": prompt_type}
    doc_ref.set(data)
    logger.info(f"Transcription saved successfully for room: {room_id}")

# Save audio buffer as a WAV file
async def save_audio(audiobuffer, room_url: str):
    if audiobuffer.has_audio():
        merged_audio = audiobuffer.merge_audio_buffers()
        filename = os.path.join(FILES_DIR, f"audio_{(urlparse(room_url).path).removeprefix('/')}.wav")
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(audiobuffer._sample_rate)
            wf.writeframes(merged_audio)
        logger.info(f"Merged audio saved to {filename}")
    else:
        logger.warning("No audio data to save")


# Main execution function
async def main(room_url: str, token: str, config_b64: str):
    # Decode the configuration
    config_str = base64.b64decode(config_b64).decode()
    config = json.loads(config_str)

    # Initialize Daily transport
    transport = DailyTransport(
        room_url,
        token,
        config['avatar_name'],
        DailyParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            camera_out_enabled=False,
            vad_enabled=True,
            vad_audio_passthrough=True,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
            transcription_settings=DailyTranscriptionSettings(language="en", tier="nova", model="2-general")
        ),
    )

    # TTS parameters setup
    # tts_params = CartesiaTTSService.InputParams(speed="normal", emotion=["positivity:high", "curiosity"])

    # Initialize TTS service and LLM service
    tts = ElevenLabsTTSService(api_key=os.getenv("ELEVENLABS_API_KEY"), voice_id=config['voice_id'])
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    # Initial messages for the chatbot
    messages = [{"role": "system", "content": config['prompt']}]

    # Initialize context and pipeline components
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)
    audiobuffer = AudioBufferProcessor()

    # Create pipeline
    pipeline = Pipeline([
        transport.input(),
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        audiobuffer,
        context_aggregator.assistant(),
    ])

    # Initialize pipeline task
    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

    # Event handler when first participant joins
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([LLMMessagesFrame(messages)])
        logger.info(f"First participant joined: {participant['id']}")

    # Event handler when participant leaves
    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        participant_id = participant['id']
        logger.info(f"Participant left: {participant_id}")

        # Save audio and transcription data, then end the pipeline task
        await save_audio(audiobuffer, room_url)
        await save_in_db((urlparse(room_url).path).removeprefix('/'), context.get_messages())
        await task.queue_frame(EndFrame())

    # Run the pipeline task
    runner = PipelineRunner()
    await runner.run(task)

# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot")
    parser.add_argument("-u", required=True, type=str, help="Room URL")
    parser.add_argument("-t", required=True, type=str, help="Token")
    parser.add_argument("--config", required=True, help="Base64 encoded configuration")
    args = parser.parse_args()

    asyncio.run(main(args.u, args.t, args.config))
