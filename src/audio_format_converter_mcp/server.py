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
from typing import Optional, Dict, Any, Tuple, Annotated, Sequence
import traceback
import logging
import sys
import base64
import wave
import audioop

import fastmcp
import fastmcp.server

from fastmcp.tools import Tool
# from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP, Context
from mcp.types import BlobResourceContents, EmbeddedResource
from pydantic import BaseModel, FileUrl, Field
from fastapi import Request, Depends
import requests
from urllib.parse import urljoin, urlparse
from fastmcp.server.dependencies import get_http_headers, get_http_request

from fastmcp.server.middleware import Middleware, MiddlewareContext

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


# Output format support
SUPPORTED_OUTPUT_FORMATS = ("wav", "mp3", "flac", "ogg")
OUTPUT_MIME_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
}


def normalize_output_format(output_format: Optional[str]) -> Optional[str]:
    """Normalize and validate the requested output format."""
    if not output_format:
        return None
    normalized = output_format.strip().lower()
    if normalized.startswith("."):
        normalized = normalized[1:]
    if normalized in SUPPORTED_OUTPUT_FORMATS:
        return normalized
    return None


def build_converted_filename(filename: str, output_format: str) -> str:
    """Build a converted filename with the requested output format."""
    stem = Path(filename).stem or "converted"
    return f"converted_{stem}.{output_format}"


# class AudioConversionResponse(BaseModel):
#     """Response model for audio conversion."""
#     success: bool
#     data: Optional[str] = None  # Base64 encoded converted audio data
#     original_info: Optional[AudioInfo] = None
#     converted_info: Optional[AudioInfo] = None
#     conversion_performed: bool = False
#     error_message: str = ""


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


def get_audio_info(audio_segment, format: str = "wav") -> AudioInfo:
    """Extract audio information from audio segment."""
    return AudioInfo(
        channels=audio_segment.channels,
        frame_rate=audio_segment.frame_rate,
        sample_width=audio_segment.sample_width,
        duration_ms=len(audio_segment),
        format=format
    )


# @mcp.tool()
# def convert_to_mono_wav(filename : str, audio_data_base64: str, target_sample_rate: int = 16000, target_sample_width: int = 2) -> Tuple[str, list]:
#     logger.info(f"Starting audio format conversion {audio_data_base64}")
#     """
#     Convert audio data to mono-channel WAV format with specified parameters.
#
#     Args:
#         filename (str): Name of file to be converted
#         audio_data_base64 (str): Base64 encoded audio data
#         target_sample_rate (int): Target sample rate in Hz (default: 16000)
#         target_sample_width (int): Target sample width in bytes (default: 2 for 16-bit)
#
#     Returns:
#         Dict[str, Any]: Response containing success status, converted audio data,
#                        original and converted audio info, and error message if failed
#     """
#     try:
#         logger.info("Starting audio format conversion")
#
#         # Decode base64 audio data
#         try:
#             audio_data = base64.b64decode(audio_data_base64)
#             logger.info(f"Successfully decoded {len(audio_data)} bytes from base64")
#         except Exception as e:
#             error_msg = f"Failed to decode base64 audio data: {e}"
#             logger.error(error_msg)
#             return AudioConversionResponse(
#                 success=False,
#                 error_message=error_msg
#             ).dict()
#
#         if len(audio_data) == 0:
#             error_msg = "Decoded audio data is empty"
#             logger.error(error_msg)
#             return AudioConversionResponse(
#                 success=False,
#                 error_message=error_msg
#             ).dict()
#
#         # Try to load audio with pydub first, then fallback to built-in
#         audio = None
#         original_info = None
#
#         try:
#             from pydub import AudioSegment
#             logger.info("Attempting to load audio with pydub")
#
#             # Write to temporary file for pydub
#             with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#                 temp_file.write(audio_data)
#                 temp_path = temp_file.name
#
#             try:
#                 audio = AudioSegment.from_file(temp_path)
#                 logger.info("Successfully loaded audio file with pydub")
#                 original_info = AudioInfo(
#                     channels=audio.channels,
#                     frame_rate=audio.frame_rate,
#                     sample_width=audio.sample_width,
#                     duration_ms=len(audio),
#                     format="detected"
#                 )
#             finally:
#                 os.unlink(temp_path)
#
#         except ImportError:
#             logger.warning("pydub not available, trying built-in WAV processing")
#         except Exception as e:
#             logger.warning(f"pydub failed to load audio: {e}, trying built-in WAV processing")
#
#         # Fallback to built-in WAV processing if pydub failed
#         if audio is None:
#             try:
#                 logger.info("Attempting built-in WAV processing")
#                 audio = load_wav_with_builtin(audio_data)
#                 logger.info("Successfully loaded audio with built-in WAV processing")
#                 original_info = get_audio_info(audio)
#             except Exception as e:
#                 error_msg = f"Both pydub and built-in WAV processing failed: {e}"
#                 logger.error(error_msg)
#                 logger.error(f"Full traceback: {traceback.format_exc()}")
#                 return AudioConversionResponse(
#                     success=False,
#                     error_message=error_msg
#                 ).dict()
#
#         logger.info(f"Original audio format - Channels: {audio.channels}, Frame rate: {audio.frame_rate}, Sample width: {audio.sample_width}, Duration: {len(audio)}ms")
#
#         # Track if any conversion was performed
#         conversion_performed = False
#
#         # Convert to mono if needed
#         if audio.channels > 1:
#             logger.info(f"Converting from {audio.channels} channels to mono")
#             audio = audio.set_channels(1)
#             conversion_performed = True
#         else:
#             logger.info("Audio is already mono")
#
#         # Set target sample rate
#         if audio.frame_rate != target_sample_rate:
#             logger.info(f"Converting sample rate from {audio.frame_rate}Hz to {target_sample_rate}Hz")
#             audio = audio.set_frame_rate(target_sample_rate)
#             conversion_performed = True
#         else:
#             logger.info(f"Audio is already at target sample rate ({target_sample_rate}Hz)")
#
#         # Set target sample width
#         if audio.sample_width != target_sample_width:
#             logger.info(f"Converting sample width from {audio.sample_width} bytes to {target_sample_width} bytes")
#             audio = audio.set_sample_width(target_sample_width)
#             conversion_performed = True
#         else:
#             logger.info(f"Audio is already at target sample width ({target_sample_width} bytes)")
#
#         converted_info = get_audio_info(audio)
#         logger.info(f"Final audio format - Channels: {audio.channels}, Frame rate: {audio.frame_rate}, Sample width: {audio.sample_width}, Duration: {len(audio)}ms")
#
#         # Export as WAV bytes
#         if hasattr(audio, 'export'):
#             # pydub AudioSegment
#             wav_data = audio.export(format="wav").read()
#         else:
#             # SimpleAudioSegment
#             wav_data = audio.export_bytes("wav")
#
#         logger.info(f"Successfully exported {len(wav_data)} bytes as WAV")
#
#         # Encode as base64 for transport
#         encoded_data = base64.b64encode(wav_data).decode('utf-8')
#
#         # return AudioConversionResponse(
#         #     success=True,
#         #     data=encoded_data,
#         #     original_info=original_info,
#         #     converted_info=converted_info,
#         #     conversion_performed=conversion_performed
#         # ).dict()
#         content_type = "audio/wav"
#         converted_filename = f"converted_{filename}"
#
#         blob = BlobResourceContents(
#             uri=FileUrl(f"file://{converted_filename}"),
#             blob=encoded_data,
#             mimeType=content_type
#         )
#
#         resource = EmbeddedResource(type="resource", resource=blob)
#         return [filename, resource]
#
#
#     except Exception as e:
#         error_msg = f"Unexpected error during audio conversion: {e}"
#         logger.error(error_msg)
#         logger.error(f"Full traceback: {traceback.format_exc()}")
#         return AudioConversionResponse(
#             success=False,
#             error_message=error_msg
#         ).dict()

def convert_audio_bytes(
    filename: str,
    audio_data: bytes,
    target_sample_rate: int = 16000,
    target_sample_width: int = 2,
    output_format: str = "wav"
) -> tuple[str, EmbeddedResource]:
    """
    Core logic for converting audio bytes to mono-channel format.

    Args:
        filename (str): Name of file to be converted
        audio_data (bytes): Raw audio data.
        target_sample_rate (int): Target sample rate in Hz.
        target_sample_width (int): Target sample width in bytes.
        output_format (str): Output audio format (e.g. wav, mp3).

    Returns:
        Tuple[str, list]: Audio conversion response.
    """
    logger.info("Starting audio format conversion from bytes")
    try:
        if len(audio_data) == 0:
            error_msg = "Audio data is empty"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            # return AudioConversionResponse(
            #     success=False,
            #     error_message=error_msg
            # ).dict()
            # return EmbeddedResource(
            #     type="resource",
            #     resource=BlobResourceContents(
            #         mimeType="text/plain",
            #         text=error_msg
            #     )
            # )

        normalized_format = normalize_output_format(output_format) or "wav"
        if normalized_format not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported output format: {output_format}")

        audio = None
        original_info = None

        try:
            from pydub import AudioSegment
            logger.info("Attempting to load audio with pydub")
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

        if audio is None:
            try:
                logger.info("Attempting built-in WAV processing")
                audio = load_wav_with_builtin(audio_data)
                logger.info("Successfully loaded audio with built-in WAV processing")
                original_info = get_audio_info(audio, format="wav")
            except Exception as e:
                error_msg = f"Both pydub and built-in WAV processing failed: {e}"
                logger.error(error_msg)
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise RuntimeError(error_msg)
                # return AudioConversionResponse(
                #     success=False,
                #     error_message=error_msg
                # ).dict()
                # return EmbeddedResource(
                #     type="resource",
                #     resource=BlobResourceContents(
                #         mimeType="text/plain",
                #         text=error_msg
                #     )
                # )

        logger.info(f"Original audio format - Channels: {audio.channels}, Frame rate: {audio.frame_rate}, Sample width: {audio.sample_width}, Duration: {len(audio)}ms")
        conversion_performed = False

        if audio.channels > 1:
            logger.info(f"Converting from {audio.channels} channels to mono")
            audio = audio.set_channels(1)
            conversion_performed = True
        else:
            logger.info("Audio is already mono")

        if audio.frame_rate != target_sample_rate:
            logger.info(f"Converting sample rate from {audio.frame_rate}Hz to {target_sample_rate}Hz")
            audio = audio.set_frame_rate(target_sample_rate)
            conversion_performed = True
        else:
            logger.info(f"Audio is already at target sample rate ({target_sample_rate}Hz)")

        if audio.sample_width != target_sample_width:
            logger.info(f"Converting sample width from {audio.sample_width} bytes to {target_sample_width} bytes")
            audio = audio.set_sample_width(target_sample_width)
            conversion_performed = True
        else:
            logger.info(f"Audio is already at target sample width ({target_sample_width} bytes)")

        converted_info = get_audio_info(audio, format=normalized_format)
        logger.info(f"Final audio format - Channels: {audio.channels}, Frame rate: {audio.frame_rate}, Sample width: {audio.sample_width}, Duration: {len(audio)}ms")

        if hasattr(audio, 'export'):
            converted_data = audio.export(format=normalized_format).read()
        else:
            converted_data = audio.export_bytes(normalized_format)

        logger.info(f"Successfully exported {len(converted_data)} bytes as {normalized_format}")
        encoded_data = base64.b64encode(converted_data).decode('utf-8')

        # return AudioConversionResponse(
        #     success=True,
        #     data=encoded_data,
        #     original_info=original_info,
        #     converted_info=converted_info,
        #     conversion_performed=conversion_performed
        # ).dict()
        content_type = OUTPUT_MIME_TYPES.get(normalized_format, "application/octet-stream")
        converted_filename = build_converted_filename(filename, normalized_format)

        blob = BlobResourceContents(
            uri=FileUrl(f"file://{converted_filename}"),
            blob=encoded_data,
            mimeType=content_type
        )

        resource = EmbeddedResource(type="resource", resource=blob)
        return (filename, resource)

    except Exception as e:
        error_msg = f"Unexpected error during audio conversion: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise RuntimeError(error_msg)
        # return AudioConversionResponse(
        #     success=False,
        #     error_message=error_msg
        # ).dict()
        # return EmbeddedResource(
        #     type="resource",
        #     resource=BlobResourceContents(
        #         mimeType="text/plain",
        #         text=error_msg
        #     )
        # )


@mcp.tool(
    name="convert_to_mono_wav",
    description="Convert base64-encoded audio data to a mono audio file suitable for speech recognition."
)
async def convert_to_mono_wav(
    ctx: Context,
    audio_data: str,
    filename: str,
    target_sample_rate: int = 16000,
    target_sample_width: int = 2,
    output_format: Optional[str] = None
) -> tuple[str, EmbeddedResource]:
    """
    Convert audio data (base64-encoded) to mono-channel format.

    Args:
        audio_data (str): Base64-encoded audio data.
        filename (str): Name of file to be converted
        target_sample_rate (int): Target sample rate in Hz (default: 16000).
        target_sample_width (int): Target sample width in bytes (default: 2 for 16-bit).
        output_format (Optional[str]): Output audio format (e.g. wav, mp3).

    Returns:
        Tuple[str, list]: Response containing
    """
    selected_format = normalize_output_format(output_format)
    if selected_format is None:
        result = await ctx.elicit(
            message="Choose output audio format",
            response_type=list(SUPPORTED_OUTPUT_FORMATS),
        )
        if result.action != "accept":
            raise RuntimeError("Output format selection was not provided.")
        selected_format = normalize_output_format(str(result.data))
        if selected_format is None:
            raise RuntimeError("Unsupported output format selected.")

    try:
        logger.info(f"Convert audio data (base64-encoded) to mono-channel format: {audio_data}")
        audio_data_raw = base64.b64decode(audio_data)
        logger.info(f"FileSize to convert:{len(audio_data_raw)}")
        with open("/tmp/output.wav", "wb") as f:
            f.write(audio_data_raw)
    except Exception as e:
        error_msg = f"Failed to decode base64 audio data: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
        # return AudioConversionResponse(
        #     success=False,
        #     error_message=error_msg
        # ).dict()
    return convert_audio_bytes(
        filename,
        audio_data_raw,
        target_sample_rate,
        target_sample_width,
        output_format=selected_format
    )

def get_base_url():
    return os.environ.get("CORE_BASE_URL", "https://statgpt-test.imf-eid.projects.epam.com/v1/")

def is_absolute_url(url):
    return bool(urlparse(url).netloc)

# @mcp.tool(
#     name="convert_uri_to_mono_wav",
#     description="Fetch an audio file from a given URI and convert it to a mono WAV format optimized for speech recognition."
# )
# def convert_uri_to_mono_wav(
#     audio_uri: str,
#     target_sample_rate: int = 16000,
#     target_sample_width: int = 2
# ) -> Dict[str, Any]:
#     """
#     Download audio from URI and convert to mono-channel WAV format.
#
#     Args:
#         audio_uri (str): URL to download the audio file from.
#         target_sample_rate (int): Target sample rate in Hz (default: 16000).
#         target_sample_width (int): Target sample width in bytes (default: 2 for 16-bit).
#
#     Returns:
#         Dict[str, Any]: Response containing:
#             - success (bool): Whether conversion was successful
#             - data (str, optional): Base64-encoded converted audio data
#             - original_info (AudioInfo, optional): Original audio format information
#             - converted_info (AudioInfo, optional): Converted audio format information
#             - conversion_performed (bool): Whether any conversion was necessary
#             - error_message (str): Error description if conversion failed
#     """
#     import requests
#     try:
#         logger.info(f"Downloading audio file from URI: {audio_uri}")
#         response = requests.get(audio_uri)
#         response.raise_for_status()
#         audio_data = response.content
#     except Exception as e:
#         error_msg = f"Failed to download audio from URI: {e}"
#         logger.error(error_msg)
#         return AudioConversionResponse(
#             success=False,
#             error_message=error_msg
#         ).dict()
#     return convert_audio_bytes(audio_data, target_sample_rate, target_sample_width)

class ConfigurationMiddleware(Middleware):
    """
    Middleware to extract and validate configuration from _meta.ai_dial_config.

    This middleware:
    1. Extracts _meta.ai_dial_config from every incoming request (tool calls and list_tools)
    2. Parses the JSON value
    3. Validates it against TextClassificationConfig model
    4. For list_tools: Dynamically injects tool_description into tool description
    5. For tool calls: Stores validated config in context for tool access
    6. Returns error if ai_dial_config is missing or invalid (for tool calls only)
    """

    # def _extract_config_from_request(self, request_data: dict) -> Optional[dict]:
    #     """Helper method to extract ai_dial_config from request data."""
    #     config_json = None
    #     if isinstance(request_data, dict):
    #         # Check for _meta in params (MCP protocol structure)
    #         if 'params' in request_data and isinstance(request_data['params'], dict):
    #             params = request_data['params']
    #             if '_meta' in params and isinstance(params['_meta'], dict):
    #                 meta = params['_meta']
    #                 if 'ai_dial_config' in meta:
    #                     config_json = meta['ai_dial_config']
    #                     logger.debug("ConfigurationMiddleware: Found ai_dial_config in params._meta")
    #
    #         # Also check for _meta at top level (alternative structure)
    #         elif '_meta' in request_data and isinstance(request_data['_meta'], dict):
    #             meta = request_data['_meta']
    #             if 'ai_dial_config' in meta:
    #                 config_json = meta['ai_dial_config']
    #                 logger.debug("ConfigurationMiddleware: Found ai_dial_config in top-level _meta")
    #
    #     return config_json

    # def _parse_and_validate_config(self, config_json) -> tuple[Optional[TextClassificationConfig], Optional[dict]]:
    #     """
    #     Helper method to parse and validate configuration.
    #
    #     Returns:
    #         tuple: (config, error_dict) where error_dict is None if successful
    #     """
    #     if config_json is None:
    #         return None, None
    #
    #     # Parse JSON if it's a string
    #     parsed_json = config_json
    #     if isinstance(config_json, str):
    #         try:
    #             parsed_json = json.loads(config_json)
    #             logger.debug("ConfigurationMiddleware: Parsed JSON string from ai_dial_config")
    #         except json.JSONDecodeError as e:
    #             logger.error(f"ConfigurationMiddleware: Invalid JSON in ai_dial_config: {e}")
    #             return None, {
    #                 "error": "Invalid JSON in ai_dial_config",
    #                 "message": f"Failed to parse JSON: {str(e)}"
    #             }
    #
    #     # Validate against the configuration model
    #     try:
    #         config = TextClassificationConfig(**parsed_json)
    #         logger.info(f"ConfigurationMiddleware: Validated configuration - endpoint: {config.hf_endpoint}, model: {config.model_name}")
    #         return config, None
    #     except ValidationError as e:
    #         logger.error(f"ConfigurationMiddleware: Validation error: {e}")
    #         return None, {
    #             "error": "Invalid configuration",
    #             "message": f"Configuration validation failed: {e.errors()}",
    #             "schema": get_config_schema()
    #         }

    async def on_list_tools(self, context: MiddlewareContext, call_next) -> Sequence[Tool]:
        """
        Intercept list_tools requests to dynamically inject tool_description from config.

        If _meta.ai_dial_config contains tool_description, it will be used to
        override the tool's description in the response.
        """
        logger.debug("ConfigurationMiddleware: Processing list_tools request")

        try:
            # Get the HTTP request to access the request body
            request = get_http_request()

            # Parse the JSON request body
            request_data = await request.json()

            # Extract _meta.ai_dial_config
            # config_json = self._extract_config_from_request(request_data)
            config_json = request_data

            # Get the original tool list
            tools = await call_next(context)

            # If configuration is provided and contains tool_description, transform the tool
            if config_json:
                config, _ = self._parse_and_validate_config(config_json)

                if config and config.tool_description:
                    logger.info(f"ConfigurationMiddleware: Injecting tool_description into list_tools response")

                    # Find the classify_text tool and transform it with the new description
                    transformed_tools = []
                    for tool in tools:
                        # if tool.name == "classify_text":
                        #     # Create a transformed version with the new description
                        #     transformed_tool = Tool.from_tool(
                        #         tool,
                        #         description=config.tool_description
                        #     )
                        #     transformed_tools.append(transformed_tool)
                        #     logger.debug(f"ConfigurationMiddleware: Transformed tool '{tool.name}' with custom description")
                        # else:
                            transformed_tools.append(tool)

                    return transformed_tools

            # Return original tools if no configuration or no tool_description
            return tools

        except RuntimeError as e:
            # get_http_request() may not be available in all contexts
            logger.warning(f"ConfigurationMiddleware: get_http_request() not available in list_tools: {e}")
            # Return original tools
            return await call_next(context)
        except Exception as e:
            logger.error(f"ConfigurationMiddleware: Unexpected error in list_tools: {type(e).__name__}: {e}", exc_info=True)
            # Return original tools on error
            return await call_next(context)

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Extract and validate configuration from _meta.ai_dial_config."""
        logger.debug("ConfigurationMiddleware: Processing tool call")

        try:
            # Get the HTTP request to access the request body
            request = get_http_request()

            # Parse the JSON request body
            request_data = await request.json()

            # Extract _meta.ai_dial_config using helper method
            config_json = self._extract_config_from_request(request_data)

            # Check if configuration is provided
            if config_json is None:
                logger.error("ConfigurationMiddleware: Missing _meta.ai_dial_config")
                # Store None to indicate missing config - tool will handle error
                if hasattr(context, 'fastmcp_context'):
                    context.fastmcp_context.set_state("config", None)
                    context.fastmcp_context.set_state("config_error", {
                        "error": "Missing configuration",
                        "message": "Please provide _meta.ai_dial_config in the request. Use GET /configuration to get the schema."
                    })
                return await call_next(context)

            # Parse and validate configuration using helper method
            config, config_error = self._parse_and_validate_config(config_json)

            if config is None:
                # Invalid configuration - store error
                if hasattr(context, 'fastmcp_context'):
                    context.fastmcp_context.set_state("config", None)
                    if config_error:
                        context.fastmcp_context.set_state("config_error", config_error)
                    else:
                        # Fallback error if parsing returned None without error details
                        context.fastmcp_context.set_state("config_error", {
                            "error": "Invalid configuration",
                            "message": "Failed to parse or validate configuration"
                        })
                return await call_next(context)

            # Store validated configuration in context
            if hasattr(context, 'fastmcp_context'):
                context.fastmcp_context.set_state("config", config)
                context.fastmcp_context.set_state("config_error", None)
                logger.debug("ConfigurationMiddleware: Stored configuration in context")

        except RuntimeError as e:
            # get_http_request() may not be available in all contexts
            logger.warning(f"ConfigurationMiddleware: get_http_request() not available: {e}")
            # Continue without configuration extraction
        except Exception as e:
            logger.error(f"ConfigurationMiddleware: Unexpected error: {type(e).__name__}: {e}", exc_info=True)
            # Continue and let the tool handle the missing config

        # Continue with the request
        return await call_next(context)


# Add the middleware to the server
# mcp.add_middleware(ConfigurationMiddleware())

@mcp.tool(
    name="convert_uri_to_mono_wav",
    # description="Fetch an audio file from a given URI and convert it to a mono WAV format optimized for speech recognition."
)
async def convert_uri_to_mono_wav(
        ctx: Context,
        audio_uri: Annotated[
            str,
            Field(
                title="DIAL URI to audio file",
                description="DIAL URI to audio file",
                json_schema_extra={"dial_url": True},
            ),
        ],
        target_sample_rate: int = 16000,
        target_sample_width: int = 2,
        output_format: Optional[str] = None,
) -> tuple[str, EmbeddedResource]:
    """
    Download audio from DIAL URI and convert to mono-channel format.
    Logs all incoming HTTP headers, passes Authorization header to download,
    and prepends CORE_BASE_URL if URI is not absolute.
    """
    selected_format = normalize_output_format(output_format)
    if selected_format is None:
        result = await ctx.elicit(
            message="Choose output audio format",
            response_type=list(SUPPORTED_OUTPUT_FORMATS),
        )
        if result.action != "accept":
            raise RuntimeError("Output format selection was not provided.")
        selected_format = normalize_output_format(str(result.data))
        if selected_format is None:
            raise RuntimeError("Unsupported output format selected.")

    try:
        # Extract API key from X-API-KEY header
        headers = get_http_headers()
        api_key = headers.get("api-key")

        # 1. Log all incoming HTTP headers
        logger.info("Incoming HTTP headers:")
        for k, v in headers.items():
            logger.info(f"  {k}: {v}")

        # 2. Pass Authorization header if present
        new_headers = {}
        if "api-key" in headers:
            new_headers["api-key"] = headers["api-key"]
            logger.info("Passing Authorization header to download request.")

        # 3. Prepend CORE_BASE_URL if URI is not absolute
        base_url = get_base_url()
        if not is_absolute_url(audio_uri) and base_url:
            full_uri = urljoin(base_url, audio_uri)
            logger.info(f"Prepended CORE_BASE_URL: {base_url} + {audio_uri} -> {full_uri}")
        else:
            full_uri = audio_uri

        logger.info(f"Downloading audio file from URI: {full_uri}")
        response = requests.get(full_uri, headers=new_headers)
        response.raise_for_status()
        audio_data = response.content
    except Exception as e:
        error_msg = f"Failed to download audio from URI: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
        # return AudioConversionResponse(
        #     success=False,
        #     error_message=error_msg
        # ).dict()
    return convert_audio_bytes(
        "converted",
        audio_data,
        target_sample_rate,
        target_sample_width,
        output_format=selected_format
    )

# @mcp.tool()
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
        raise RuntimeError(error_msg)


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