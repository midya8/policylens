(function(root){
'use strict';
const sum=xs=>{let s=0,c=0;for(const x of xs){const t=s+x;c+=Math.abs(s)>=Math.abs(x)?(s-t)+x:(x-t)+s;s=t;}return s+c;};
function evaluate(p,cap=null){
 if(cap!==null&&(!Number.isFinite(cap)||cap<=0))throw Error('Cap must be finite and positive.');
 if(!p||Object.keys(p).sort().join()!=['action_names','rows'].join()||!Array.isArray(p.rows)||!p.rows.length||p.rows.length>256||!Array.isArray(p.action_names)||p.action_names.length<2||p.action_names.length>16||new Set(p.action_names).size!==p.action_names.length||p.action_names.some(x=>typeof x!=='string'||!x.trim()))throw Error('Invalid bounded audit input.');
 const n=p.rows.length,d=p.action_names.length,ws=[],ds=[],ips=[],dr=[];
 for(const r of p.rows){
  if(Object.keys(r).sort().join()!==['action','behavior','reward','reward_model','target'].join()||!Number.isInteger(r.action)||r.action<0||r.action>=d||!Number.isFinite(r.reward)||r.reward<0||r.reward>1)throw Error('Invalid row.');
  for(const v of [r.behavior,r.target,r.reward_model])if(!Array.isArray(v)||v.length!==d||v.some(x=>typeof x!=='number'||!Number.isFinite(x)||x<0||x>1))throw Error('Invalid vector.');
  if(Math.abs(sum(r.behavior)-1)>1e-9||Math.abs(sum(r.target)-1)>1e-9||r.behavior[r.action]<=0)throw Error('Invalid propensity distribution.');
  let w=r.target[r.action]/r.behavior[r.action];if(!Number.isFinite(w))throw Error('Weight exceeds finite range.');if(cap!==null)w=Math.min(w,cap);
  const dm=sum(r.target.map((v,j)=>v*r.reward_model[j]));ws.push(w);ds.push(dm);ips.push(w*r.reward);dr.push(dm+w*(r.reward-r.reward_model[r.action]));
 }
 const max=Math.max(...ws),scaled=ws.map(w=>max?w/max:0),mass=sum(scaled),unsupported=p.rows.filter(r=>r.target.some((t,j)=>t>0&&r.behavior[j]===0)).length;
 const out={n,ips:sum(ips.map(x=>x/n)),snips:mass?sum(scaled.map((s,i)=>s*p.rows[i].reward))/mass:null,direct_method:sum(ds.map(x=>x/n)),doubly_robust:sum(dr.map(x=>x/n)),clip:cap,weights:{mean:sum(ws.map(x=>x/n)),max,effective_sample_size:mass?mass*mass/sum(scaled.map(x=>x*x)):0,zero_fraction:ws.filter(w=>w===0).length/n},support:{rows_with_unsupported_target_actions:unsupported,observed_context_support_complete:unsupported===0},interpretation:unsupported?'support_violation_no_nonparametric_identification':cap!==null?'clipped_estimands_may_be_biased':'assumptions_still_required'};
 for(const v of [out.ips,out.snips,out.direct_method,out.doubly_robust,out.weights.mean,out.weights.effective_sample_size])if(v!==null&&!Number.isFinite(v))throw Error('Estimate exceeds finite range.');return out;
}
function audit(p,cap=null){
 const raw=evaluate(p),selected=evaluate(p,cap),n=p.rows.length;
 const ws=p.rows.map(r=>r.target[r.action]/r.behavior[r.action]),used=ws.map(w=>cap===null?w:Math.min(w,cap)),max=Math.max(...used),ss=used.map(w=>max?w/max:0),mass=sum(ss);
 const rows=p.rows.map((r,i)=>{const a=r.action,w=ws[i],u=used[i],dm=sum(r.target.map((t,j)=>t*r.reward_model[j])),residual=r.reward-r.reward_model[a],normalized=mass?ss[i]/mass:null;return{row:i,action:a,reward:r.reward,behavior_observed:r.behavior[a],target_observed:r.target[a],raw_weight:w,used_weight:u,clipped:u<w,direct_row:dm,residual,ips_contribution:u*r.reward/n,direct_contribution:dm/n,dr_contribution:(dm+u*residual)/n,normalized_weight:normalized,snips_contribution:normalized===null?null:normalized*r.reward,ips_removed_by_cap:(w-u)*r.reward/n,dr_removed_by_cap:(w-u)*residual/n,unsupported_actions:r.target.flatMap((t,j)=>t>0&&r.behavior[j]===0?[j]:[])};});
 const loo=n<2?[]:p.rows.map((r,i)=>{const x=evaluate({action_names:p.action_names,rows:p.rows.filter((_,j)=>j!==i)},cap);return{removed_row:i,n:n-1,ips:x.ips,snips:x.snips,direct_method:x.direct_method,doubly_robust:x.doubly_robust};});
 return {raw,selected,rows,leave_one_row_out:loo,cap_breakpoints:[...new Set(ws.filter(w=>w>0))].sort((a,b)=>a-b)};
}
function edited(c,mix=1,index=0,reward=null){if(!Number.isFinite(mix)||mix<0||mix>1||!Number.isInteger(index)||index<0||index>=c.input.rows.length||reward!==null&&(!Number.isFinite(reward)||reward<0||reward>1))throw Error('Invalid edit.');const p=JSON.parse(JSON.stringify(c.input));for(const r of p.rows)r.target=r.target.map((t,j)=>(1-mix)*r.behavior[j]+mix*t);if(reward!==null)p.rows[index].reward=reward;return p;}
const api={sum,evaluate,audit,edited};if(typeof module!=='undefined')module.exports=api;root.PolicyLedger=api;
})(typeof window!=='undefined'?window:globalThis);
