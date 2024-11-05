<script>
    import { onMount } from "svelte";

    let peerConnection;
    let videoRef;

    async function connectWebSocket() {
        const socket = new WebSocket("ws://localhost:8000/ws");

        // Wait for the WebSocket connection to open
        socket.onopen = async () => {
            console.log("WebSocket connected");

            // Set up WebRTC connection
            peerConnection = new RTCPeerConnection();
            peerConnection.onicecandidate = ({ candidate }) => {
                if (candidate) {
                    socket.send(JSON.stringify({ type: "ice", candidate }));
                }
            };

            // Handle incoming video stream
            peerConnection.ontrack = (event) => {
                videoRef.srcObject = event.streams[0];
            };

            // Create WebRTC offer after WebSocket is open
            const offer = await peerConnection.createOffer();
            await peerConnection.setLocalDescription(offer);

            // Send the offer through WebSocket
            socket.send(JSON.stringify({ type: "offer", sdp: offer.sdp }));
        };

        socket.onclose = () => console.log("WebSocket disconnected");

        socket.onmessage = async (event) => {
            const message = JSON.parse(event.data);
            if (message.type === "answer") {
                await peerConnection.setRemoteDescription(new RTCSessionDescription(message));
            } else if (message.type === "ice") {
                const candidate = new RTCIceCandidate(message.candidate);
                await peerConnection.addIceCandidate(candidate);
            }
        };
    }

    onMount(() => {
        connectWebSocket();
    });
</script>

<video bind:this={videoRef} autoplay playsinline></video>
<style>
    video {
        width: 100%;
        height: auto;
    }
</style>
