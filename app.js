// Pixel 9 Local AI Lab v2.0 — browser-only. Transformers.js is loaded lazily.
const $ = id => document.getElementById(id);
const MB = 1024 * 1024;
const CHUNK_MIB = 32;
const CHECKPOINT_KEY = 'pixel9-local-ai-lab-v2-memory-checkpoint';
const LIBRARY_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';
const MODELS = {
  small: {id:'onnx-community/SmolLM2-135M-Instruct-ONNX-MHA',gpu:'q4f16',cpu:'q8'},
  smol360: {id:'onnx-community/SmolLM2-360M-Instruct-ONNX',gpu:'q4f16',cpu:'q8'},
  lfm350: {id:'onnx-community/LFM2.5-350M-ONNX',gpu:'q4',cpu:'q8'},
  gemma1b: {id:'onnx-community/gemma-3-1b-it-ONNX',gpu:'q4f16',cpu:'q8'},
  lfm12: {id:'onnx-community/LFM2-1.2B-ONNX',gpu:'q4',cpu:'q8'},
  smol17: {id:'HuggingFaceTB/SmolLM2-1.7B-Instruct',gpu:'q4',cpu:'q8'},
};
const state = {adapter:null, features:[], records:[], log:[], busy:false, memRunning:false, stopRequested:false, pipe:null, loaded:null, loadMs:null, pipelineLib:null};
const when = () => new Date().toISOString();
const show = (id,value) => {$(id).textContent=String(value);};
const fmtS = ms => Number.isFinite(ms)?`${(ms/1000).toFixed(2)} s`:'—';
const fmtGiB = n => Number.isFinite(n)?`${(n/1024**3).toFixed(2)} GiB`:'—';
const escapeCsv = v => `"${String(v == null?'':typeof v==='object'?JSON.stringify(v):v).replaceAll('"','""')}"`;
function log(text) {
  const entry=`${new Date().toLocaleTimeString()} ${text}`;
  state.log.push(entry); if (state.log.length>100) state.log.shift();
  show('log',state.log.join('\n'));
}
function setModelStatus(text){show('model-status',text);log(text);}
function setMemStatus(text){show('mem-status',text);log(text);}
function refreshControls(){
  const modelBusy=state.busy;
  $('load').disabled=modelBusy || state.memRunning;
  $('generate').disabled=modelBusy || state.memRunning || !state.pipe;
  $('unload').disabled=modelBusy || state.memRunning || !state.pipe;
  $('mem-start').disabled=modelBusy || state.memRunning || !!state.pipe;
  $('mem-stop').disabled=!state.memRunning;
  for(const id of ['model','backend','dtype'])$(id).disabled=modelBusy || state.memRunning || !!state.pipe;
  $('memlimit').disabled=state.memRunning || state.busy || !!state.pipe;
  $('rescan').disabled=state.memRunning || modelBusy;
}
function addRecord(record){
  state.records.push(record);
  const tbody=$('results');
  if(state.records.length===1)tbody.replaceChildren();
  const tr=document.createElement('tr');
  for (const v of [record.type,record.outcome,record.type==='model'?fmtS(record.generation_ms):fmtS(record.duration_ms),record.summary||'']){
    const td=document.createElement('td');td.textContent=String(v);td.className='out';tr.append(td);
  }
  tbody.prepend(tr);
}
async function probeGPU(){
  show('secure',window.isSecureContext?'Yes':'No — use HTTPS');
  show('ram',navigator.deviceMemory?`${navigator.deviceMemory} GB (rounded)`:'Not exposed');
  show('browser',(navigator.userAgent.match(/(?:Chrome|CriOS)\/[^ ;]+/)||[])[0]||'Other');
  if(!navigator.gpu){state.adapter=null;show('webgpu','Unavailable');show('adapter','—');show('f16','—');show('adapter-debug','navigator.gpu missing.');return;}
  try{
    state.adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
    if(!state.adapter) throw new Error('Adapter request returned null.');
    const a=state.adapter;
    state.features=Array.from(a.features).sort();
    const info=a.info||{};
    show('webgpu','Available');show('adapter',[info.vendor,info.architecture].filter(Boolean).join(' / ')||'Detected');
    show('f16',a.features.has('shader-f16')?'Supported':'Not exposed');
    show('maxbuffer',fmtGiB(a.limits.maxBufferSize));
    show('maxbinding',fmtGiB(a.limits.maxStorageBufferBindingSize));
    show('adapter-debug',JSON.stringify({info,features:state.features,limits:{maxBufferSize:a.limits.maxBufferSize,maxStorageBufferBindingSize:a.limits.maxStorageBufferBindingSize,maxComputeInvocationsPerWorkgroup:a.limits.maxComputeInvocationsPerWorkgroup}},null,2));
    log('WebGPU adapter available.');
  }catch(e){state.adapter=null;show('webgpu','Error');show('adapter-debug',String(e));log(`WebGPU probe error: ${e}`);}
  refreshControls();
}
// GPU clear touches the full buffer; nonzero sample writes and readback confirm the
// queue completed. This tests cumulative live GPU buffers, not physical VRAM capacity.
async function testChunk(device,seed){
  const bytes=CHUNK_MIB*MB;
  let buf=null,read=null,errors=[];
  device.pushErrorScope('out-of-memory');device.pushErrorScope('validation');
  try{
    buf=device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    const pages=bytes/4096;
    const readPages=[0,Math.floor(pages/4),Math.floor(pages/2),Math.floor(3*pages/4),pages-1];
    read=device.createBuffer({size:readPages.length*256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const clear=device.createCommandEncoder();
    clear.clearBuffer(buf);
    device.queue.submit([clear.finish()]);
    for(const p of readPages){
      const expected=((seed>>>0)^p^0x9e3779b9)>>>0;
      device.queue.writeBuffer(buf,p*4096,new Uint32Array([expected]));
    }
    const verify=device.createCommandEncoder();
    readPages.forEach((p,i)=>verify.copyBufferToBuffer(buf,p*4096,read,i*256,4));
    device.queue.submit([verify.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const array=new DataView(read.getMappedRange());
    readPages.forEach((p,i)=>{
      const actual=array.getUint32(i*256,true);
      const expected=((seed>>>0)^p^0x9e3779b9)>>>0;
      if(actual!==expected)throw new Error(`Readback mismatch at page ${p}: expected ${expected}, got ${actual}`);
    });
  }catch(e){errors.push(String(e?.message||e));}
  finally{
    if(read){try{if(read.mapState==='mapped')read.unmap();}catch(_){ }read.destroy();}
    for(const name of ['validation','out-of-memory']){
      try{const e=await device.popErrorScope();if(e)errors.push(`${name}: ${e.message}`);}catch(e){errors.push(`error scope: ${e.message}`);}
    }
  }
  if(errors.length){if(buf)buf.destroy();throw new Error(errors.join('; '));}
  return buf;
}
$('mem-start').addEventListener('click',async()=>{
  if(state.memRunning||state.busy||state.pipe)return;
  if(!state.adapter){setMemStatus('No WebGPU adapter available.');return;}
  state.memRunning=true;state.stopRequested=false;refreshControls();
  const cap=Number($('memlimit').value),start=performance.now();
  if(cap>1024 && !confirm(`High-risk memory test: ${cap} MiB cumulative. Chrome or Android may terminate the tab; any current session results can be lost. Continue?`)){state.memRunning=false;refreshControls();return;}
  checkpoint({status:'running',target_mib:cap,largest_verified_mib:0,updated_at:when()});
  let device=null,held=[],maxMiB=0,reason='Target reached',deviceLost=false;
  show('mem-pass','0 MiB');show('mem-result','Running');$('mem-progress').max=cap;$('mem-progress').value=0;
  try{
    device=await state.adapter.requestDevice();
    device.lost.then(info=>{if(state.memRunning){deviceLost=true;log(`GPU device lost: ${info.message||info.reason}`);}});
    for(let allocation=CHUNK_MIB;allocation<=cap;allocation+=CHUNK_MIB){
      if(state.stopRequested){reason='Stopped by user';break;}
      if(deviceLost){reason='GPU device lost';break;}
      const chunk=await testChunk(device,(0x8ace0000+allocation)>>>0);
      held.push(chunk);maxMiB=allocation;
      $('mem-progress').value=allocation;show('mem-pass',`${allocation} MiB`);
      show('mem-status',`Verified ${allocation} / ${cap} MiB in ${held.length} buffers; holding until test ends.`);
      checkpoint({status:'running',target_mib:cap,largest_verified_mib:maxMiB,updated_at:when()});
      // Yield to update progress and allow Stop to register. Allocate sequentially.
      await new Promise(resolve=>setTimeout(resolve,40));
    }
  }catch(e){reason=`Failed: ${e?.message||e}`;log(reason);}
  finally{
    held.forEach(b=>{try{b.destroy();}catch(_){}});held=[];
    if(device){try{device.destroy();}catch(_){}}
    state.memRunning=false;refreshControls();
    show('mem-result',reason);checkpoint({status:reason==='Target reached'?'complete':reason==='Stopped by user'?'stopped':'failed',target_mib:cap,largest_verified_mib:maxMiB,reason,updated_at:when()});
    setMemStatus(`Finished. Largest verified cumulative allocation: ${maxMiB} MiB. Buffers released.`);
    addRecord({type:'memory',timestamp:when(),outcome:reason==='Target reached'?'PASS':reason==='Stopped by user'?'STOP':'FAIL',
      target_mib:cap,largest_verified_mib:maxMiB,chunk_mib:CHUNK_MIB,method:'32 MiB STORAGE buffers cleared by GPU, five nonzero probes per buffer verified by GPU-to-CPU readback',
      reason,duration_ms:performance.now()-start,summary:`${maxMiB} / ${cap} MiB; ${reason}`,gpu_info:gpuInfo()});
  }
});
$('mem-stop').addEventListener('click',()=>{state.stopRequested=true;show('mem-status','Stop requested. Finishing current chunk, then releasing buffers…');});
function checkpoint(obj){
  try{localStorage.setItem(CHECKPOINT_KEY,JSON.stringify(obj));show('mem-checkpoint',`Saved checkpoint: ${obj.largest_verified_mib||0} MiB · ${obj.status}`);}catch(e){log(`Checkpoint storage unavailable: ${e.message}`);}
}
function restoreCheckpoint(){
  try{const raw=localStorage.getItem(CHECKPOINT_KEY);if(!raw)return;const c=JSON.parse(raw);
    if(c.status==='running')c.status='interrupted (tab/browser ended before cleanup)';
    show('mem-checkpoint',`Previous memory test: ${c.largest_verified_mib||0} / ${c.target_mib||0} MiB · ${c.status} · ${c.updated_at||''}`);
    if(c.status.startsWith('interrupted'))log('Previous memory test may have been interrupted. This is not proof of an OOM threshold.');
  }catch(e){log(`Previous checkpoint unreadable: ${e.message}`);}
}
function gpuInfo(){const a=state.adapter;return a?{info:a.info||{},shader_f16:a.features.has('shader-f16'),max_buffer_bytes:a.limits.maxBufferSize,max_storage_binding_bytes:a.limits.maxStorageBufferBindingSize}:null;}
function chooseModel(){
  const model=MODELS[$('model').value];const backend=$('backend').value;
  const selected=$('dtype').value;
  return {id:model.id,backend,dtype:selected==='auto'?(backend==='webgpu'?model.gpu:model.cpu):selected};
}
async function unload(){
  if(state.pipe){try{await state.pipe.dispose?.();}catch(e){log(`Dispose: ${e}`);}}
  state.pipe=null;state.loaded=null;state.loadMs=null;
  setModelStatus('Model unloaded. For the cleanest comparison, reload the page between large models.');refreshControls();
}
$('load').addEventListener('click',async()=>{
  if(state.busy||state.memRunning)return;
  state.busy=true;refreshControls();
  const selected=chooseModel();
  try{
    if(selected.backend==='webgpu'&&!state.adapter)throw new Error('No WebGPU adapter; select WebAssembly.');
    if(selected.dtype==='q4f16'&&selected.backend==='webgpu'&&!state.features.includes('shader-f16'))
      log('Note: shader-f16 not exposed; q4f16 may fail. Try q4 for this model if available.');
    setModelStatus(`Importing Transformers.js 4.3.2; ${selected.id} (${selected.backend}, ${selected.dtype})…`);
    const started=performance.now();
    if(!state.pipelineLib)state.pipelineLib=await import(LIBRARY_URL);
    const {pipeline}=state.pipelineLib;
    $('model-progress').hidden=false;
    const p=await pipeline('text-generation',selected.id,{
      device:selected.backend,dtype:selected.dtype,
      progress_callback:info=>{
        if(info.status==='progress'&&Number.isFinite(info.progress)){
          $('model-progress').value=info.progress;
          show('model-status',`Loading ${info.file||'model'}: ${info.progress.toFixed(1)}%`);
        } else if(info.status==='initiate') show('model-status',`Fetching ${info.file||'model files'}…`);
        else if(info.status==='done') show('model-status',`Loaded ${info.file||'a file'}…`);
      }
    });
    state.pipe=p;state.loaded=selected;state.loadMs=performance.now()-started;
    show('load-ms',fmtS(state.loadMs));setModelStatus(`Ready: ${selected.id} (${selected.backend}/${selected.dtype})`);
  }catch(e){state.pipe=null;state.loaded=null;setModelStatus(`Model load failed: ${e?.message||e}`);
    addRecord({type:'load',timestamp:when(),outcome:'FAIL',model:selected.id,backend:selected.backend,dtype:selected.dtype,reason:String(e?.message||e),summary:selected.id});
  }finally{$('model-progress').hidden=true;state.busy=false;refreshControls();}
});
$('generate').addEventListener('click',async()=>{
  if(!state.pipe||state.busy)return;
  const prompt=$('prompt').value.trim();if(!prompt){setModelStatus('Enter a prompt.');return;}
  state.busy=true;refreshControls();
  const maxTokens=Number($('max-tokens').value);let answer='';let firstMs=null;const start=performance.now();
  try{
    setModelStatus('Generating locally…');
    const {TextStreamer}=state.pipelineLib;
    const streamer=new TextStreamer(state.pipe.tokenizer,{
      skip_prompt:true,skip_special_tokens:true,
      callback_function:chunk=>{if(firstMs===null&&String(chunk).trim().length)firstMs=performance.now()-start;}
    });
    const result=await state.pipe([{role:'user',content:prompt}],{max_new_tokens:maxTokens,do_sample:false,streamer});
    const elapsed=performance.now()-start;
    const generated=result?.[0]?.generated_text;
    answer=Array.isArray(generated)?String(generated.filter(m=>m.role==='assistant').at(-1)?.content??''):String(generated??JSON.stringify(result));
    let tokens=null;
    try{
      const encoded=state.pipe.tokenizer(answer,{add_special_tokens:false});
      const ids=encoded.input_ids;
      tokens=ids?.size??ids?.data?.length??(Array.isArray(ids)?ids.length:null);
    }catch(_){ /* Output token estimate unavailable */ }
    const tokps=Number.isFinite(tokens)?tokens/(elapsed/1000):null;
    show('answer',answer||'(empty response)');show('first-ms',fmtS(firstMs));show('generation-ms',fmtS(elapsed));
    show('tokps',Number.isFinite(tokps)?`${tokps.toFixed(2)} tok/s`:'Unavailable');
    addRecord({type:'model',timestamp:when(),outcome:'PASS',model:state.loaded.id,backend:state.loaded.backend,dtype:state.loaded.dtype,
      model_load_ms:state.loadMs,first_stream_text_ms:firstMs,generation_ms:elapsed,retokenised_output_tokens:tokens,
      full_wall_tokens_per_s:tokps,max_new_tokens:maxTokens,prompt,output:answer,summary:`${state.loaded.id}: ${tokps?.toFixed(2)??'?'} tok/s`});
    setModelStatus('Generation completed. Try a second prompt or export results.');
  }catch(e){setModelStatus(`Generation failed: ${e?.message||e}`);
    addRecord({type:'model',timestamp:when(),outcome:'FAIL',model:state.loaded?.id,backend:state.loaded?.backend,
      dtype:state.loaded?.dtype,prompt,max_new_tokens:maxTokens,reason:String(e?.message||e),summary:'Generation error'});
  }finally{state.busy=false;refreshControls();}
});
$('unload').addEventListener('click',unload);
function exportPayload(){return {schema:'pixel9-local-ai-lab-v2',created_at:when(),user_agent:navigator.userAgent,
  device_memory_gb_reported:navigator.deviceMemory??null,gpu:gpuInfo(),records:state.records,events:state.log,
  caveats:['Memory test does not report total GPU VRAM or guaranteed capacity','Generation tok/s uses total wall time and retokenised output']};}
function download(name,data,type){const blob=new Blob([data],{type});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),4000);}
$('json').addEventListener('click',()=>download('pixel9_ai_lab_results.json',JSON.stringify(exportPayload(),null,2),'application/json'));
$('csv').addEventListener('click',()=>{
  const cols=['timestamp','type','outcome','model','backend','dtype','target_mib','largest_verified_mib','model_load_ms','first_stream_text_ms','generation_ms','retokenised_output_tokens','full_wall_tokens_per_s','prompt','output','reason'];
  const rows=[cols.join(','),...state.records.map(r=>cols.map(c=>escapeCsv(r[c])).join(','))];
  download('pixel9_ai_lab_results.csv',rows.join('\r\n'),'text/csv');
});
$('copy').addEventListener('click',async()=>{
  const lines=[`Pixel 9 Local AI Lab v2 (${when()})`,`WebGPU: ${$('adapter').textContent}; shader-f16: ${$('f16').textContent}`,
    ...state.records.map(r=>r.type==='memory'?`Memory ${r.outcome}: ${r.largest_verified_mib}/${r.target_mib} MiB (${r.reason})`:
    `${r.type} ${r.outcome}: ${r.model??''} ${r.backend??''} ${r.dtype??''} ${r.full_wall_tokens_per_s?.toFixed(2)??'-'} tok/s`)];
  try{await navigator.clipboard.writeText(lines.join('\n'));log('Summary copied to clipboard.');}
  catch(e){log(`Clipboard unavailable: ${e.message}; use JSON export instead.`);}
});
$('rescan').addEventListener('click',probeGPU);
function showPwaState(){
  const installed=matchMedia('(display-mode: standalone)').matches;
  const cached=!!navigator.serviceWorker?.controller;
  show('pwa', `${installed?'Installed':'Browser'} · ${cached?'Shell active':'Shell pending'} · ${navigator.onLine?'Online':'Offline'}`);
}
if('serviceWorker' in navigator&&location.protocol==='https:'){
  window.addEventListener('load',()=>navigator.serviceWorker.register('./sw.js')
    .then(showPwaState).catch(e=>log(`PWA shell cache unavailable: ${e.message}`)));
  navigator.serviceWorker.addEventListener('controllerchange',showPwaState);
}
window.addEventListener('online',showPwaState);
window.addEventListener('offline',showPwaState);
showPwaState();
refreshControls();restoreCheckpoint();probeGPU();
