'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const A=require('../docs/ledger.js');require('../docs/data.js');
const root=path.join(__dirname,'../policylens/artifacts');
function near(a,b){if(typeof a==='number'&&typeof b==='number'){assert.ok(Number.isFinite(a)&&Number.isFinite(b));assert.ok(Math.abs(a-b)<=1e-12+1e-12*Math.abs(b),`${a} vs ${b}`);}else assert.deepEqual(a,b);}
function compare(a,b){for(const k of Object.keys(a)){if(k==='scope')continue;if(a[k]&&typeof a[k]==='object'){assert.ok(b[k]);if(Array.isArray(a[k])){assert.equal(a[k].length,b[k].length);a[k].forEach((v,i)=>typeof v==='object'&&v!==null?compare(v,b[k][i]):near(v,b[k][i]));}else compare(a[k],b[k]);}else near(a[k],b[k]);}}
function witnesses(){return POLICY_DATA.cases.flatMap(c=>fs.readdirSync(root).filter(f=>f.startsWith('ledger-'+c.id+'-')).sort().flatMap(f=>JSON.parse(fs.readFileSync(path.join(root,f)))));}
module.exports={witnesses,compare};
test('independent JS agrees with95 native edit ledgers including full LOO',()=>{const w=witnesses();assert.equal(w.length,95);for(const e of w){const c=POLICY_DATA.cases.find(c=>c.id===e.case);compare(A.audit(A.edited(c,e.mix,e.row,e.reward),e.cap),e.audit);}});
test('browser validation, zero SNIPS, support and negative DR remain explicit',()=>{for(const cap of [0,-1,NaN,Infinity])assert.throws(()=>A.audit(POLICY_DATA.cases[0].input,cap));assert.equal(A.audit(POLICY_DATA.cases[2].input).selected.snips,null);assert.ok(A.audit(POLICY_DATA.cases[3].input).selected.doubly_robust<0);assert.equal(A.audit(POLICY_DATA.cases[1].input).selected.support.observed_context_support_complete,false);for(const mix of [-1,2,NaN])assert.throws(()=>A.edited(POLICY_DATA.cases[0],mix));});
test('independent integer quarter-grid oracle80two-row games',()=>{
 // Rational targets/rewards/q with behavior one-half. Exact common denominators.
 for(let k=0;k<80;k++){
  const rows=[],nums=[],ds=[],dr=[],weights=[];
  for(let j=0;j<2;j++){const t=(k+j*3)%5,r=(k*3+j)%5,q0=(k+j)%5,q1=(k*2+j)%5,a=(k+j)%2;const w=a?4-t:t;weights.push(w);nums.push(w*r);const d=t*q0+(4-t)*q1;ds.push(d);dr.push(d+2*w*(r-(a?q1:q0)));rows.push({action:a,reward:r/4,behavior:[.5,.5],target:[t/4,1-t/4],reward_model:[q0/4,q1/4]});}
  const x=A.audit({action_names:['a','b'],rows}).selected;
  near(x.ips,(nums[0]+nums[1])/16);near(x.direct_method,(ds[0]+ds[1])/32);near(x.doubly_robust,(dr[0]+dr[1])/32);
  if(weights[0]+weights[1])near(x.snips,(nums[0]+nums[1])/(4*(weights[0]+weights[1])));else assert.equal(x.snips,null);
 }
});
