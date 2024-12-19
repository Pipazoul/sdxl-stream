<script lang="ts">
    let prompt = "A cat";
    let socketUrl = `${import.meta.env.VITE_API_WEBSOCKET}/ws/update_config`;
    let ws = new WebSocket(socketUrl);
    function sendConfig(live: boolean) {
        if (live && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                prompt: prompt
            }));
        }
    }

    $: {
        sendConfig(true);
    }
</script>

<section>
    <div style="display: flex; flex-direction: column; height: 100vh;">
        <h1 class="uppercase text-6xl text-white">Prompt</h1>
        <br>
        <textarea 
            bind:value={prompt} 
            on:input={() => sendConfig(true)} 
            class="bg-gray-800 text-white text-2xl uppercase"
            style="flex-grow: 1; width: 100%; font-size: 2xl; padding: 10px; box-sizing: border-box;" 
        ></textarea>
    </div>
</section>
