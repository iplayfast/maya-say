#!/usr/bin/env python3
"""
Socket protocol for maya-say TTS communication.
Handles multiplexed message and audio data streaming.
"""

import json
import struct
import socket
from typing import Dict, Any, Optional, Iterator


class SocketProtocol:
    """Protocol for sending/receiving messages and audio over sockets."""

    # Command types
    CMD_JSON = b"JSON"      # JSON message
    CMD_AUDIO = b"AUDI"     # Binary audio data chunk
    CMD_END = b"DONE"       # End of stream
    CMD_ERROR = b"ERRO"     # Error message

    HEADER_SIZE = 8  # 4 bytes command + 4 bytes length

    @staticmethod
    def send_json(sock: socket.socket, data: Dict[str, Any]) -> None:
        """Send a JSON message over the socket."""
        json_bytes = json.dumps(data).encode('utf-8')
        header = SocketProtocol.CMD_JSON + struct.pack('I', len(json_bytes))
        sock.sendall(header + json_bytes)

    @staticmethod
    def send_audio(sock: socket.socket, audio_data: bytes) -> None:
        """Send an audio data chunk over the socket."""
        header = SocketProtocol.CMD_AUDIO + struct.pack('I', len(audio_data))
        sock.sendall(header + audio_data)

    @staticmethod
    def send_end(sock: socket.socket) -> None:
        """Send end-of-stream marker."""
        header = SocketProtocol.CMD_END + struct.pack('I', 0)
        sock.sendall(header)

    @staticmethod
    def send_error(sock: socket.socket, error_message: str) -> None:
        """Send an error message."""
        error_bytes = error_message.encode('utf-8')
        header = SocketProtocol.CMD_ERROR + struct.pack('I', len(error_bytes))
        sock.sendall(header + error_bytes)

    @staticmethod
    def recv_exact(sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        data = b''
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Socket connection closed")
            data += chunk
        return data

    @staticmethod
    def receive_message(sock: socket.socket) -> tuple[str, Any]:
        """
        Receive a message from the socket.
        Returns: (message_type, data)
        - message_type: 'json', 'audio', 'end', or 'error'
        - data: dict for json, bytes for audio/error, None for end
        """
        # Read header
        header = SocketProtocol.recv_exact(sock, SocketProtocol.HEADER_SIZE)
        command = header[:4]
        length = struct.unpack('I', header[4:8])[0]

        # Read payload if present
        payload = b''
        if length > 0:
            payload = SocketProtocol.recv_exact(sock, length)

        # Parse based on command type
        if command == SocketProtocol.CMD_JSON:
            return ('json', json.loads(payload.decode('utf-8')))
        elif command == SocketProtocol.CMD_AUDIO:
            return ('audio', payload)
        elif command == SocketProtocol.CMD_END:
            return ('end', None)
        elif command == SocketProtocol.CMD_ERROR:
            return ('error', payload.decode('utf-8'))
        else:
            raise ValueError(f"Unknown command type: {command}")

    @staticmethod
    def receive_stream(sock: socket.socket) -> Iterator[tuple[str, Any]]:
        """
        Receive a stream of messages until END marker.
        Yields: (message_type, data) tuples
        """
        while True:
            msg_type, data = SocketProtocol.receive_message(sock)
            yield (msg_type, data)
            if msg_type == 'end' or msg_type == 'error':
                break
