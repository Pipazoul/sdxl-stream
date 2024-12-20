<script lang="ts">
    let apiUrl = `${import.meta.env.VITE_API_DOMAIN}`;
    let socketUrl = `${import.meta.env.VITE_API_WEBSOCKET}/ws/update_config`;
    console.log("socketUrl",socketUrl);
    let ws = new WebSocket(socketUrl);
    let prompt = "A cat";
    let width = 512;
    let height = 512;
    let guidance_scale = 0;
    let controlnet_conditioning_scale = 0.8;
    let realtime = true;
    let seed = -1;
    let camera = false;
    let videoSource: HTMLVideoElement;
    let cameraCanvas: HTMLCanvasElement;
    let currentCameraIndex = 0;

    let streamPreview = false;
    let availableDevices: MediaDeviceInfo[] = [];
    let currentStream: MediaStream | null = null;

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
            seed = Math.floor(Math.random() * 1000000);
            // Start capturing frames every 5 seconds if camera mode is active
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
        realtime = false;
        realtime = true;
        seed = -1;
    };

    function sendConfig(live: boolean) {
        if (live && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                prompt: prompt,
                width: width,
                height: height,
                guidance_scale: guidance_scale,
                controlnet_conditioning_scale: controlnet_conditioning_scale,
                seed: seed
            }));
        }
    }

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
                prompt: prompt,
                width: width,
                height: height,
                guidance_scale: guidance_scale,
                seed: seed,
                base64_image: base64Image.split(",")[1]  // Remove the data URL prefix
            }));
        }
    }

    $: {
        if (realtime) {
            sendConfig(true);
        }
    }
</script>

<section>
    <!--Mjpeg feed https://back-socket.quantic.homes/video_feed-->
    {#if streamPreview}
    <div>
        <h2>MJPEG Preview</h2>
        <img src="{apiUrl}/video_feed" alt="MJPEG Feed" style="width: 100%; max-width: 512px; border: 1px solid #ccc;" />
    </div>
    {/if}
    <h1>Config</h1>
    <p>Config page content</p>
    <div>
        <input type="checkbox" bind:checked={realtime} class="checkbox" />
        <label for="realtime">Realtime</label>
    </div>
    <div>
        <label for="seed">Seed</label>
        <input type="number" bind:value={seed} on:input={() => sendConfig(realtime)} />
    </div>
    <div>
        <label for="prompt">Prompt</label>
        <input type="text" bind:value={prompt} on:input={() => sendConfig(realtime)} />
    </div>
    <div>
        <label for="width">Width {width}</label>
        <input type="range" min="256" max="1024" step="64" bind:value={width} class="range range-primary" on:input={() => sendConfig(realtime)} />
    </div>
    <div>
        <label for="height">Height {height}</label>
        <input type="range" min="256" max="1024" step="64" bind:value={height} class="range range-primary" on:input={() => sendConfig(realtime)} />
    </div>
    <div>
        <label for="guidance_scale">Guidance Scale</label>
        <input type="range" min="0" max="10" step="1" bind:value={guidance_scale} class="range range-primary" on:input={() => sendConfig(realtime)} />
    </div>
    <div>
        <label for="guidance_scale">Controlnet conditonning Scale</label>
        <input type="range" min="0" max="1" step="0.1" bind:value={controlnet_conditioning_scale} class="range range-primary" on:input={() => sendConfig(realtime)} />
    </div>
    <div>
        <input type="checkbox" bind:checked={streamPreview} class="checkbox" />
        <label for="streamPreview">Stream Preview</label>
    </div>
    <div>
        <video bind:this={videoSource}></video>
        <button class="btn" on:click={obtenerVideoCamara}>Enable Camera</button>
        <button class="btn" on:click={switchCamera}>Switch Camera</button>
        <button class="btn" on:click={stopCamera}>Stop Camera</button>
        <canvas bind:this={cameraCanvas} style="display:none;"></canvas>
    </div>
</section>
