import { readFileSync, writeFileSync, appendFileSync, realpathSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { loadConfig } from 'file:///D:/Github/agent-collab/src/config.ts';
import { createAdapter } from 'file:///D:/Github/agent-collab/src/adapter-factory.ts';
const engine=process.argv[2];
if(!['dsh','zcode'].includes(engine)) throw Error('Expected dsh or zcode');
const root=realpathSync('D:/Github/new-viewer');
const out=root+'/docs/rebuild-20261106';
const config=loadConfig('D:/Github/agent-collab/config.local.json');
const agent=config.agents.find(a=>a.id===engine&&a.engine===engine);
if(!agent) throw Error('Configured agent not found');
const adapter=createAdapter(agent);
const startedAt=new Date().toISOString();
const controller=new AbortController();
const timeout=setTimeout(()=>controller.abort(),900000);
const events=out+'/'+engine+'-events.jsonl';
const safeEvents=new Set(['adapter.mode','session.started','session.mode.confirmed','agent.turn.completed','process.started']);
const focus=engine==='dsh'?'重点质询算法语义、统计假设、GPU物理计价和反例。':'重点质询架构边界、迁移路线、测试独立性、结果契约及UI联动。';
try{
 console.log(JSON.stringify({engine,startedAt,access:'read',transport:'existing-agent-collab-native-adapter'}));
 const result=await adapter.run({taskId:randomUUID(),runId:randomUUID(),goal:'独立质询统一推理仿真重构方案',instructions:readFileSync(out+'/review-packet.md','utf8')+'\n'+focus,directory:root,access:'read',dependencies:[],deadlineAt:Date.now()+900000,requestApproval:async()=> 'deny'},controller.signal,(type,payload)=>{if(safeEvents.has(type))appendFileSync(events,JSON.stringify({at:new Date().toISOString(),type,payload})+'\n');});
 writeFileSync(out+'/'+engine+'-questions.md',result.text);
 writeFileSync(out+'/'+engine+'-receipt.json',JSON.stringify({engine,startedAt,endedAt:new Date().toISOString(),success:true,sessionId:result.sessionId,access:'read'},null,2));
 console.log(JSON.stringify({engine,success:true,chars:result.text.length}));
}catch(e){const error=e instanceof Error?e.message:String(e);writeFileSync(out+'/'+engine+'-receipt.json',JSON.stringify({engine,startedAt,endedAt:new Date().toISOString(),success:false,error},null,2));console.log(JSON.stringify({engine,success:false,error}));process.exitCode=1;}
finally{clearTimeout(timeout);await adapter.close?.();}
