import { readFileSync, writeFileSync, appendFileSync, realpathSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { loadConfig } from 'file:///D:/Github/agent-collab/src/config.ts';
import { createAdapter } from 'file:///D:/Github/agent-collab/src/adapter-factory.ts';
const engine=process.argv[2];
if(!['dsh','zcode'].includes(engine)) throw Error('Expected dsh or zcode');
const root=realpathSync('D:/Github/new-viewer');
const out=root+'/docs/rebuild-20261106/review-ui-round2';
const config=loadConfig('D:/Github/agent-collab/config.local.json');
const agent=config.agents.find(a=>a.id===engine&&a.engine===engine);
if(!agent) throw Error('Configured agent not found');
const adapter=createAdapter(agent);
const startedAt=new Date().toISOString();
const controller=new AbortController();
const timeout=setTimeout(()=>controller.abort(),900000);
const events=out+'/'+engine+'-events.jsonl';
const safeEvents=new Set(['adapter.mode','session.started','session.mode.confirmed','agent.turn.completed','process.started']);
const focus=engine==='dsh'?'你的问题 ID 使用 D2-01 等。侧重可辨识性、统计/物理口径、独立正确性反例，同时完整检查页面。':'你的问题 ID 使用 Z2-01 等。侧重页面遗漏、协议字段充分性、聚合/联动/重放及验证盲区，同时检查指标数学。';
try{
 console.log(JSON.stringify({engine,startedAt,access:'read',transport:'existing-agent-collab-native-adapter'}));
 const result=await adapter.run({taskId:randomUUID(),runId:randomUUID(),goal:'第二轮独立复核 UI 数据契约和验证充分性',instructions:readFileSync(out+'/packet.md','utf8')+'\n'+focus,directory:root,access:'read',dependencies:[],deadlineAt:Date.now()+900000,requestApproval:async()=> 'deny'},controller.signal,(type,payload)=>{if(safeEvents.has(type))appendFileSync(events,JSON.stringify({at:new Date().toISOString(),type,payload})+'\n');});
 writeFileSync(out+'/'+engine+'-review.md',result.text);
 writeFileSync(out+'/'+engine+'-receipt.json',JSON.stringify({engine,startedAt,endedAt:new Date().toISOString(),success:true,sessionId:result.sessionId,access:'read'},null,2));
 console.log(JSON.stringify({engine,success:true,chars:result.text.length}));
}catch(e){const error=e instanceof Error?e.message:String(e);writeFileSync(out+'/'+engine+'-receipt.json',JSON.stringify({engine,startedAt,endedAt:new Date().toISOString(),success:false,error},null,2));console.log(JSON.stringify({engine,success:false,error}));process.exitCode=1;}
finally{clearTimeout(timeout);await adapter.close?.();}
