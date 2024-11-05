<script lang="ts">
    let socketUrl = `${import.meta.env.VITE_SOCKET}/ws/update_config`;
    let ws = new WebSocket(socketUrl);
    let prompt = "A cat";
    let width = 512;
    let height = 512;
    let guidance_scale = 0;
    let realtime = false;
    let seed = -1;

    ws.onopen = () => {
        console.log("WebSocket connection established");
    };

    function sendConfig(live: boolean) {
        if (live && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                prompt: prompt,
                width: width,
                height: height,
                guidance_scale: guidance_scale,
                seed: seed
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
        <input type="range" min="256" max="1024" step="64" bind:value={width} class="range range-primary"  on:input={() => sendConfig(realtime)} />
    </div>
    <div>
        <label for="height">Height {height}</label>
        <input type="range" min="256" max="1024" step="64" bind:value={height} class="range range-primary" on:input={() => sendConfig(realtime)} />
    </div>
    <div>
        <label for="guidance_scale">Guidance Scale</label>
        <input type="range" min="0" max="10" step="1" bind:value={guidance_scale} class="range range-primary" on:input={() => sendConfig(realtime)} />
    </div>
</section>
