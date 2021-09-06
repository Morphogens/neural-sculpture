<script lang="ts">
    import * as d3 from 'd3';
    import { onMount } from 'svelte';

    export let points: number[] = [1, 4, 6, 2, 6, 2, 6, 2];
    let clientWidth: number;
    let clientHeight: number;
    let svgElement: SVGElement;
    type D3Select = ReturnType<typeof d3.select>;
    let path: D3Select;
    let svg: D3Select;
    onMount(() => {
        svg = d3.select(svgElement);
        path = svg.append("path");
    });
    
    $: xScale = d3.scaleLinear().domain([0, points.length]).range([0, clientWidth]);
    $: yScale = d3.scaleLinear().domain([Math.min(...points), Math.max(...points)]).range([clientHeight, 0]);
    
    $:{
        const min = Math.min(...points);
        const max = Math.max(...points);
        const line = d3.line<number>()
            .x((d, i) => xScale(i))
            .y(d => yScale(d))
            .curve(d3.curveCatmullRom.alpha(.5))

        if (path) {
            svg.select("path")
              .datum(points)
              .attr("d", line)
                .attr("stroke", "black")
                .attr("stroke-width", 2)
                .attr("fill", "none")
                // .attr("transform", null)
                //   .transition()
                //  .duration(100)
                // .attr("transform", "translate(" + xScale(-1) + ",0)")

        }
    }


</script>
{#if points.length}
    <div class="label">{points[points.length - 1]}</div>
{/if}
<div bind:clientHeight bind:clientWidth style="width: 200px; height: 200px; background: #ddd;">
    <svg bind:this={svgElement} width={clientWidth} height={clientHeight} >
    </svg>
</div>

<style>
    .label {
        background: white;
        padding: 4px;
    }
</style>