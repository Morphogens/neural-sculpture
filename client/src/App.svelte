<script lang="ts">
import { onMount } from 'svelte'
import * as THREE from 'three'
import { debounce } from 'ts-debounce'
import setupScene from './sceneSetup'
import LossHistory from "./LossHistory.svelte"
import { socket, socketOpen, messageServer } from './stores/socket'
import { mode } from './stores/state'
import { lossStore, meshStore as mesh } from './stores/mesh'
import { sculpControls, optPositionPanel } from './stores/panels'
const MAX_RES = 64

////////////////////////////////////////////////////////////////////////////////
let mouseDown = false
let mouseClicked = false
let shiftDown = false
let meshScale = 1.0
const mouse = new THREE.Vector2();
const lastCamera = new THREE.Vector3()
let intersectPoint: null | THREE.Vector3 = null

////////////////////////////////////////////////////////////////////////////////
const { scene, raycaster, camera, controls, sceneUpdate } = setupScene()
const geometry = new THREE.SphereGeometry( 2.0, 8, 8 );
const meshMaterial = new THREE.MeshPhongMaterial({
    opacity: .5,
    transparent: true,
    color: 0x156289,
    emissive: 0x072534,
    side: THREE.DoubleSide,
    flatShading: true
})
const sphere = new THREE.Mesh( geometry, meshMaterial )
scene.add( sphere )

////////////////////////////////////////////////////////////////////////////////
// Server communication.
const sendNewCamera = debounce(() => {
    messageServer('camera', camera.position.toArray())
}, 200)

const submitSettings = debounce(() => {
    const settings = {
        ...$sculpControls,
        point: (sphere.position.clone().divideScalar(meshScale)).toArray(),
        optimize_radius: $optPositionPanel.optimize_radius,
    }
    console.log('Sending sculp_settings', settings);
    messageServer("sculp_settings", settings)
}, 100);

////////////////////////////////////////////////////////////////////////////////
// Listen for new meshes from the server.
let lastMesh = null
mesh.subscribe($mesh => {
    if (lastMesh == null){
        messageServer("initialize", "cat.npy")
    }
    if (lastMesh) {
        scene.remove(lastMesh)
    }
    if ($mesh) {
        meshScale = MAX_RES / $sculpControls.grid_resolution
        $mesh.scale.set(meshScale, meshScale, meshScale)
        scene.add($mesh)
        console.log('Got new mesh', {meshScale})
        lastMesh = $mesh
    }
})

////////////////////////////////////////////////////////////////////////////////
// Detect which position is being moused over.
let mouseOverMesh = false
function detectMeshIntersection() {
    if ($mesh) {
        raycaster.setFromCamera( mouse, camera );
        const intersects = raycaster.intersectObjects($mesh.children);
        if (intersects.length) {
            mouseOverMesh = true
            intersectPoint = intersects[0].point
            if ($mode == 'setting-sphere') {
                sphere.position.copy(intersectPoint)
            }
        } else {
            intersectPoint = null
            mouseOverMesh = false
        }
    }
}
const sphereColors = {
    'inactive': {r: 0.2, g: 0.2, b: 0.2},
    'positive': {r: 0.1, g: 1.0, b: 0.5372549019607843, a: .5},
    'negative': {r: 0.9, g: 0.3843137254901961, b: 0.5372549019607843},
}
$: sphere.visible = $mode == 'default' || mouseOverMesh
// $: sphereColorName = mouseDown ? (shiftDown ? 'negative' : 'positive') : 'inactive'
// $: sphere.material.color= sphereColors[sphereColorName]
sphere.material.color = sphereColors['positive']
$: sphere.material.opacity = $optPositionPanel.view_sphere ? ($mode == 'setting-sphere' ? .5 : .8) : 0
$: sphere.scale.set(
    $optPositionPanel.optimize_radius * .5,
    $optPositionPanel.optimize_radius * .5,
    $optPositionPanel.optimize_radius * .5
)

////////////////////////////////////////////////////////////////////////////////
// Event listeners
function resetClicked() {
    messageServer("initialize", "12140_Skull_v3_L2.npy")
}

function onKeyDown(e: KeyboardEvent) {
    shiftDown = e.shiftKey;
    if (e.key.toLowerCase() === "t") {
        // console.log("SCUlPTING!")
        // if (!isSculping){
        //     isSculping = true;
        //     messageServer("sculp_mode", {is_sculping: isSculping})
        // }
        $sculpControls.sculp_enabled = !$sculpControls.sculp_enabled
    }
}

function onKeyUp(e: KeyboardEvent) {
    shiftDown = e.shiftKey;
    // if (e.key.toLowerCase() === "t") {
    //     console.log("SCUlPTING STOPPED!")
    //     isSculping = false;
    //     messageServer("stop_sculp_mode", {is_sculping: isSculping})
    // };

}

function onMouseMove(event) {
    // calculate mouse position in normalized device coordinates
    // (-1 to +1) for both components
    mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
    mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
}

function onMouseDown() {
    mouseDown = true
    if ($mode == 'setting-sphere' && mouseOverMesh) {
        $mode = 'default'
    }    
    if ($mode == 'default' && mouseOverMesh && $sculpControls.sculp_enabled) {
        $mode = 'sculpting'
    }
}

function onMouseUp() {
    $mode = 'default'
}

$: if ($mode == 'sculpting') {
    controls.enableRotate = false
    controls.enablePan = false
    document.body.style.cursor = 'pointer'
} else {
    controls.enableRotate = true
    controls.enablePan = true
    document.body.style.cursor = 'default'
}

// If the sculpting sphere changes update the server.
$: if (
    $sculpControls || 
    ($mode == 'default' && $optPositionPanel.optimize_radius)
) {
    submitSettings()
}

////////////////////////////////////////////////////////////////////////////////
// Main event loop.
function loop(): void {
    requestAnimationFrame(loop)
    detectMeshIntersection()
    if ($socketOpen && !camera.position.equals(lastCamera)) {
        sendNewCamera()
        lastCamera.copy(camera.position)
    }
    if ($mode == 'sculpting' && intersectPoint) {
        console.log('SCULPTING', {
            additive: !shiftDown,
            point: intersectPoint.toArray()
        })
        // TODO
        // messageServer('sculpt_cursor', {
        // })
    }
    sceneUpdate()
}
onMount(() => {
    loop()
})
</script>

<svelte:body
    on:mousemove={onMouseMove}
    on:mousedown={onMouseDown}
    on:mouseup={onMouseUp}
    on:click={() => mouseClicked=true}
    on:mouseup={() => mouseDown=false}
    on:keydown={onKeyDown}
    on:keyup={onKeyUp}
/>
<div id='container'>
    <div class="instructions">
        <div>instructions</div>
        <p>Press "T" to toggle sculping mode on/off. Click an area to target the sculping</p>
    </div>
    <div class="connection">
        <div class="indicator" class:live={$socketOpen} class:closed={!$socketOpen}/>Connection: {$socketOpen ? "Live" : "Disconnected" }
    </div>
    <button on:click={resetClicked}>
        <div>Reset Mesh + Optimizer</div>
         <div>(takes a few seconds)</div>
    </button>
    <LossHistory points={$lossStore["camera"]}/>
</div>

<style>
    .instructions {
        background: white;
        padding: 4px;
        margin: 1px;
    }
    .connection {
        background: white;
        display: flex;
        align-items: center;
        padding: 4px;
    }

    .sculp-mode {
        background: white;
        display: flex;
        align-items: center;
        padding: 4px;
    }
    .indicator {
        width: 10px;
        height: 10px;
        border-radius: 100%;
        margin-right: 2px;
    }
    .indicator.live {
        background: #059669;
    }
    .indicator.closed {
        background: #DC2626;
    }
    :global(body) {
        margin: 0px;
        padding: 0px;
    }
    #container {
        width: 200px;
        position: absolute;
        display: flex;
        flex-flow: column;
    }
        
    input[type=text] {
        /* width: 100%; */
        padding: 12px 20px;
        margin: 8px 0;
        box-sizing: border-box;
    }
</style>
