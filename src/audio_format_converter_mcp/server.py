#!/usr/bin/env python3
"""
Audio Format Converter MCP Server

A Model Context Protocol server that validates and converts audio files to mono format
suitable for speech recognition processing.
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import traceback
import logging
import sys
import base64
import wave
import audioop
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('audio_format_converter.log')
    ]
)
logger = logging.getLogger(__name__)

# Create the MCP server instance
mcp = FastMCP("Audio Format Converter MCP Server")


class AudioInfo(BaseModel):
    """Audio file information model."""
    channels: int
    frame_rate: int
    sample_width: int
    duration_ms: float
    format: str


class AudioConversionResponse(BaseModel):
    """Response model for audio conversion."""
    success: bool
    data: Optional[str] = None  # Base64 encoded converted audio data
    original_info: Optional[AudioInfo] = None
    converted_info: Optional[AudioInfo] = None
    conversion_performed: bool = False
    error_message: str = ""


def load_wav_with_builtin(wav_data: bytes) -> 'SimpleAudioSegment':
    """
    Load WAV file from bytes using Python's built-in libraries.
    
    Args:
        wav_data: WAV file data as bytes
        
    Returns:
        SimpleAudioSegment object with audio data and properties
    """
    logger.info("Loading WAV file with built-in libraries from bytes")
    
    # Write bytes to temporary file for wave module
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(wav_data)
        temp_path = temp_file.name
    
    try:
        with wave.open(temp_path, 'rb') as wav_file:
            # Get WAV file properties
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            
            logger.info(f"WAV properties - Channels: {channels}, Sample width: {sample_width}, Frame rate: {frame_rate}, Frames: {frames}")
            
            # Read all audio data
            raw_data = wav_file.readframes(frames)
            
        return SimpleAudioSegment(raw_data, channels, frame_rate, sample_width)
    finally:
        # Clean up temporary file
        os.unlink(temp_path)


class SimpleAudioSegment:
    """Simple audio segment class that mimics pydub's AudioSegment interface."""
    
    def __init__(self, raw_data: bytes, channels: int, frame_rate: int, sample_width: int):
        self._raw_data = raw_data
        self.channels = channels
        self.frame_rate = frame_rate
        self.sample_width = sample_width
        self._duration_ms = (len(raw_data) / (channels * sample_width * frame_rate)) * 1000
    
    def __len__(self):
        return int(self._duration_ms)
    
    def set_channels(self, new_channels: int) -> 'SimpleAudioSegment':
        """Convert audio to specified number of channels."""
        if new_channels == self.channels:
            return self
        
        if self.channels == 2 and new_channels == 1:
            # Convert stereo to mono
            logger.info("Converting stereo to mono using built-in audioop")
            mono_data = audioop.tomono(self._raw_data, self.sample_width, 1, 1)
            return SimpleAudioSegment(mono_data, 1, self.frame_rate, self.sample_width)
        else:
            raise ValueError(f"Unsupported channel conversion: {self.channels} -> {new_channels}")
    
    def set_frame_rate(self, new_rate: int) -> 'SimpleAudioSegment':
        """Convert audio to specified sample rate."""
        if new_rate == self.frame_rate:
            return self
        
        logger.info(f"Converting sample rate from {self.frame_rate}Hz to {new_rate}Hz using built-in audioop")
        converted_data, _ = audioop.ratecv(
            self._raw_data, self.sample_width, self.channels, 
            self.frame_rate, new_rate, None
        )
        return SimpleAudioSegment(converted_data, self.channels, new_rate, self.sample_width)
    
    def set_sample_width(self, new_width: int) -> 'SimpleAudioSegment':
        """Convert audio to specified sample width."""
        if new_width == self.sample_width:
            return self
        
        logger.info(f"Converting sample width from {self.sample_width} bytes to {new_width} bytes using built-in audioop")
        
        if self.sample_width == 1 and new_width == 2:
            converted_data = audioop.lin2lin(self._raw_data, 1, 2)
        elif self.sample_width == 2 and new_width == 1:
            converted_data = audioop.lin2lin(self._raw_data, 2, 1)
        elif self.sample_width == 2 and new_width == 4:
            converted_data = audioop.lin2lin(self._raw_data, 2, 4)
        elif self.sample_width == 4 and new_width == 2:
            converted_data = audioop.lin2lin(self._raw_data, 4, 2)
        else:
            raise ValueError(f"Unsupported sample width conversion: {self.sample_width} -> {new_width}")
        
        return SimpleAudioSegment(converted_data, self.channels, self.frame_rate, new_width)
    
    def export_bytes(self, format: str = "wav") -> bytes:
        """Export audio as bytes in specified format."""
        if format != "wav":
            raise ValueError(f"Built-in processor only supports WAV export, not {format}")
        
        logger.info("Exporting WAV file as bytes")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            with wave.open(temp_path, 'wb') as out_wav:
                out_wav.setnchannels(self.channels)
                out_wav.setsampwidth(self.sample_width)
                out_wav.setframerate(self.frame_rate)
                out_wav.writeframes(self._raw_data)
            
            with open(temp_path, 'rb') as f:
                return f.read()
        finally:
            os.unlink(temp_path)


def get_audio_info(audio_segment) -> AudioInfo:
    """Extract audio information from audio segment."""
    return AudioInfo(
        channels=audio_segment.channels,
        frame_rate=audio_segment.frame_rate,
        sample_width=audio_segment.sample_width,
        duration_ms=len(audio_segment),
        format="wav"
    )


@mcp.tool()
def convert_to_mono_wav(audio_data_base64: str, target_sample_rate: int = 16000, target_sample_width: int = 2) -> Dict[str, Any]:
    """
    Convert audio data to mono-channel WAV format with specified parameters.
    
    Args:
        audio_data_base64 (str): Base64 encoded audio data
        target_sample_rate (int): Target sample rate in Hz (default: 16000)
        target_sample_width (int): Target sample width in bytes (default: 2 for 16-bit)
        
    Returns:
        Dict[str, Any]: Response containing success status, converted audio data,
                       original and converted audio info, and error message if failed
    """
    try:
        logger.info("Starting audio format conversion")
        
        # Decode base64 audio data
        try:
            audio_data = base64.b64decode(audio_data_base64)
            logger.info(f"Successfully decoded {len(audio_data)} bytes from base64")
        except Exception as e:
            error_msg = f"Failed to decode base64 audio data: {e}"
            logger.error(error_msg)
            return AudioConversionResponse(
                success=False,
                error_message=error_msg
            ).dict()
        
        if len(audio_data) == 0:
            error_msg = "Decoded audio data is empty"
            logger.error(error_msg)
            return AudioConversionResponse(
                success=False,
                error_message=error_msg
            ).dict()
        
        # Try to load audio with pydub first, then fallback to built-in
        audio = None
        original_info = None
        
        try:
            from pydub import AudioSegment
            logger.info("Attempting to load audio with pydub")
            
            # Write to temporary file for pydub
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                audio = AudioSegment.from_file(temp_path)
                logger.info("Successfully loaded audio file with pydub")
                original_info = AudioInfo(
                    channels=audio.channels,
                    frame_rate=audio.frame_rate,
                    sample_width=audio.sample_width,
                    duration_ms=len(audio),
                    format="detected"
                )
            finally:
                os.unlink(temp_path)
                
        except ImportError:
            logger.warning("pydub not available, trying built-in WAV processing")
        except Exception as e:
            logger.warning(f"pydub failed to load audio: {e}, trying built-in WAV processing")
        
        # Fallback to built-in WAV processing if pydub failed
        if audio is None:
            try:
                logger.info("Attempting built-in WAV processing")
                audio = load_wav_with_builtin(audio_data)
                logger.info("Successfully loaded audio with built-in WAV processing")
                original_info = get_audio_info(audio)
            except Exception as e:
                error_msg = f"Both pydub and built-in WAV processing failed: {e}"
                logger.error(error_msg)
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return AudioConversionResponse(
                    success=False,
                    error_message=error_msg
                ).dict()
        
        logger.info(f"Original audio format - Channels: {audio.channels}, Frame rate: {audio.frame_rate}, Sample width: {audio.sample_width}, Duration: {len(audio)}ms")
        
        # Track if any conversion was performed
        conversion_performed = False
        
        # Convert to mono if needed
        if audio.channels > 1:
            logger.info(f"Converting from {audio.channels} channels to mono")
            audio = audio.set_channels(1)
            conversion_performed = True
        else:
            logger.info("Audio is already mono")
        
        # Set target sample rate
        if audio.frame_rate != target_sample_rate:
            logger.info(f"Converting sample rate from {audio.frame_rate}Hz to {target_sample_rate}Hz")
            audio = audio.set_frame_rate(target_sample_rate)
            conversion_performed = True
        else:
            logger.info(f"Audio is already at target sample rate ({target_sample_rate}Hz)")
        
        # Set target sample width
        if audio.sample_width != target_sample_width:
            logger.info(f"Converting sample width from {audio.sample_width} bytes to {target_sample_width} bytes")
            audio = audio.set_sample_width(target_sample_width)
            conversion_performed = True
        else:
            logger.info(f"Audio is already at target sample width ({target_sample_width} bytes)")
        
        converted_info = get_audio_info(audio)
        logger.info(f"Final audio format - Channels: {audio.channels}, Frame rate: {audio.frame_rate}, Sample width: {audio.sample_width}, Duration: {len(audio)}ms")
        
        # Export as WAV bytes
        if hasattr(audio, 'export'):
            # pydub AudioSegment
            wav_data = audio.export(format="wav").read()
        else:
            # SimpleAudioSegment
            wav_data = audio.export_bytes("wav")
        
        logger.info(f"Successfully exported {len(wav_data)} bytes as WAV")
        
        # Encode as base64 for transport
        encoded_data = base64.b64encode(wav_data).decode('utf-8')
        
        return AudioConversionResponse(
            success=True,
            data=encoded_data,
            original_info=original_info,
            converted_info=converted_info,
            conversion_performed=conversion_performed
        ).dict()
        
    except Exception as e:
        error_msg = f"Unexpected error during audio conversion: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return AudioConversionResponse(
            success=False,
            error_message=error_msg
        ).dict()


@mcp.tool()
def validate_audio_format(audio_data_base64: str) -> Dict[str, Any]:
    """
    Validate and analyze audio format without conversion.
    
    Args:
        audio_data_base64 (str): Base64 encoded audio data
        
    Returns:
        Dict[str, Any]: Audio format information and validation results
    """
    try:
        logger.info("Starting audio format validation")
        
        # Decode base64 audio data
        try:
            audio_data = base64.b64decode(audio_data_base64)
            logger.info(f"Successfully decoded {len(audio_data)} bytes from base64")
        except Exception as e:
            error_msg = f"Failed to decode base64 audio data: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error_message": error_msg
            }
        
        if len(audio_data) == 0:
            return {
                "success": False,
                "error_message": "Decoded audio data is empty"
            }
        
        # Try to analyze audio format
        audio_info = None
        
        try:
            from pydub import AudioSegment
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                audio = AudioSegment.from_file(temp_path)
                audio_info = AudioInfo(
                    channels=audio.channels,
                    frame_rate=audio.frame_rate,
                    sample_width=audio.sample_width,
                    duration_ms=len(audio),
                    format="detected"
                )
            finally:
                os.unlink(temp_path)
                
        except (ImportError, Exception) as e:
            logger.warning(f"pydub analysis failed: {e}, trying built-in WAV analysis")
            
            try:
                audio = load_wav_with_builtin(audio_data)
                audio_info = get_audio_info(audio)
            except Exception as e:
                return {
                    "success": False,
                    "error_message": f"Could not analyze audio format: {e}"
                }
        
        # Determine what conversions would be needed
        needs_conversion = []
        if audio_info.channels > 1:
            needs_conversion.append(f"Convert from {audio_info.channels} channels to mono")
        if audio_info.frame_rate != 16000:
            needs_conversion.append(f"Convert sample rate from {audio_info.frame_rate}Hz to 16000Hz")
        if audio_info.sample_width != 2:
            needs_conversion.append(f"Convert sample width from {audio_info.sample_width} bytes to 2 bytes")
        
        return {
            "success": True,
            "audio_info": audio_info.dict(),
            "is_mono": audio_info.channels == 1,
            "is_16khz": audio_info.frame_rate == 16000,
            "is_16bit": audio_info.sample_width == 2,
            "ready_for_asr": len(needs_conversion) == 0,
            "required_conversions": needs_conversion
        }
        
    except Exception as e:
        error_msg = f"Unexpected error during audio validation: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error_message": error_msg
        }


def setup_health_endpoint():
    """Set up health check endpoint for the FastAPI app."""
    try:
        app = mcp.streamable_http_app()
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint for Docker and monitoring systems."""
            from datetime import datetime
            return {
                "status": "healthy",
                "service": "Audio Format Converter MCP Server",
                "version": "0.1.0",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "tools": ["convert_to_mono_wav", "validate_audio_format"]
            }
        
        logger.info("Health check endpoint configured at /health")
        return app
    except Exception as e:
        logger.warning(f"Could not set up health endpoint: {e}")
        return None


def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="Audio Format Converter MCP Server")
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080, 
        help="Port to run the server on (default: 8080)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost", 
        help="Host to bind the server to (default: localhost)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting Audio Format Converter MCP Server on http://{args.host}:{args.port}")
    print(f"MCP endpoint will be available at: http://{args.host}:{args.port}/mcp")
    
    import uvicorn
    
    # Set up health endpoint and get the app
    app = setup_health_endpoint()
    if app is None:
        app = mcp.streamable_http_app()
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()