import * as knobby from 'svelte-knobby';
import { mode } from './state'
import { messageServer } from './socket'
const MAX_RES = 64

// sent to server as sculp_settings
export const sculpControls = knobby.panel({
    sculp_enabled: true,
    prompt: "A sculpture of a bunny rabbit",
    learning_rate: 1,
    batch_size: 1,
    grid_resolution: {
        $label: 'grid_resolution',
        value: MAX_RES,
        min: 16,
        max: MAX_RES,
        step: 4
    },
    reset_mesh: () => messageServer("initialize", "cat.npy"),
});

// let spherePositionSet = false
export const optPositionPanel = knobby.panel({    
    view_sphere: true,
    optimize_radius: {
        value: 8,
        min: 1,
        max: MAX_RES / 2,
        step: 1
    },
    reset_position: value => {
        mode.set('setting-sphere')
        value.view_sphere = true
        return value
    },
})