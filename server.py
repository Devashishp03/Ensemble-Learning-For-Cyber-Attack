# server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import random
import asyncio
import pyshark
import time

app = FastAPI()

# Mount static files for HTML/JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Selected features model
class TrafficData(BaseModel):
    timestamp: float
    src_ip: str
    dst_ip: str
    protocol: str
    length: int
    flags: str

# Replace generate_mock_packet() with this for real data

def generate_real_packet():
    capture = pyshark.LiveCapture(interface='eth0')
    for packet in capture.sniff_continuously(packet_count=1):
        return TrafficData(
            timestamp=float(packet.sniff_time.timestamp()),
            src_ip=packet.ip.src,
            dst_ip=packet.ip.dst,
            protocol=packet.transport_layer,
            length=int(packet.length),
            flags=getattr(packet.tcp, 'flags', '') if hasattr(packet, 'tcp') else ''
        )
# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send new packet every second (adjust as needed)
            await asyncio.sleep(1)
            packet = generate_real_packet()
            await websocket.send_json(packet.dict())
    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/scan-now")
async def manual_scan():
    """Endpoint for manual scan button"""
    return generate_real_packet()