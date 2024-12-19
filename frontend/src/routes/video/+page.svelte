<script lang="ts">
    let socketUrl = `${import.meta.env.VITE_API_WEBSOCKET}/ws/update_config`;
    let ws = new WebSocket(socketUrl);

    let camera = false;
    let videoSource: HTMLVideoElement;
    let cameraCanvas: HTMLCanvasElement;
    let currentCameraIndex = 0;
    let availableDevices: MediaDeviceInfo[] = [];
    let currentStream: MediaStream | null = null;
    let seed = Math.floor(Math.random() * 1000000);
    ws.onopen = () => {
        console.log("WebSocket connection established");
    };

    const obtenerVideoCamara = async () => {
        try {
            availableDevices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = availableDevices.filter(device => device.kind === 'videoinput');
            
            if (videoDevices.length === 0) {
                throw new Error('No video devices found.');
            }

            currentStream?.getTracks().forEach(track => track.stop());

            const stream = await navigator.mediaDevices.getUserMedia({
                video: { deviceId: videoDevices[currentCameraIndex].deviceId }
            });

            currentStream = stream;
            videoSource.srcObject = stream;
            videoSource.play();
            camera = true;

            // ws send seed random and controlnet_conditioning_scale 0.8
            ws.send(JSON.stringify({
                seed: seed,
                controlnet_conditioning_scale: 0.8
            }));

            startFrameCapture();
        } catch (error) {
            console.log(error);
        }
    };

    const switchCamera = async () => {
        try {
            const videoDevices = availableDevices.filter(device => device.kind === 'videoinput');
            if (videoDevices.length <= 1) {
                console.log("No other cameras available to switch.");
                return;
            }

            currentCameraIndex = (currentCameraIndex + 1) % videoDevices.length;
            await obtenerVideoCamara();
        } catch (error) {
            console.error("Error switching camera:", error);
        }
    };

    const stopCamera = () => {
        camera = false;
        currentStream?.getTracks().forEach(track => track.stop());
        currentStream = null;
        videoSource.srcObject = null;
        console.log("Camera stopped.");
        // send seed: -1 to ws  
        ws.send(JSON.stringify({
            seed: -1,
            controlnet_conditioning_scale: 0.0
        }));
        // Close the WebSocket connection when the camera is stopped
        ws.close();

    };

    function startFrameCapture() {
        if (camera) {
            const captureInterval = setInterval(() => {
                if (!camera) {
                    clearInterval(captureInterval);
                    return;
                }
                captureFrame();
            }, 100); // Capture every 100 milliseconds
        }
    }

    function captureFrame() {
        if (!camera || !videoSource || !cameraCanvas) return;

        cameraCanvas.width = videoSource.videoWidth;
        cameraCanvas.height = videoSource.videoHeight;
        const context = cameraCanvas.getContext("2d");
        if (context) {
            context.drawImage(videoSource, 0, 0, videoSource.videoWidth, videoSource.videoHeight);
            const imageData = cameraCanvas.toDataURL("image/jpeg");
            sendImage(imageData);
        }
    }

    function sendImage(base64Image: string) {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                base64_image: base64Image.split(",")[1]  // Remove the data URL prefix
            }));
        }
    }

    // Auto-start the camera when the script loads
    obtenerVideoCamara();
</script>

<section>
    <div>
        <video bind:this={videoSource}></video>
        <button class="btn" on:click={obtenerVideoCamara}>Enable Camera</button>
        <button class="btn" on:click={switchCamera}>Switch Camera</button>
        <button class="btn" on:click={stopCamera}>Stop Camera</button>
        <canvas bind:this={cameraCanvas} style="display:none;"></canvas>
    </div>
</section>
