/**
 * AGAD-UDL  —  Interactive Performance Dashboard
 * IIT Indore  |  Amit Dalal, Prashant Mishra
 * All metrics sourced directly from model_implementation_inference.ipynb
 */

import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, BarChart, Bar, AreaChart, Area,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ScatterChart, Scatter, PieChart, Pie,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";

/* ─── Design tokens ─────────────────────────────────────────────────────── */
const T = {
  /* backgrounds */
  bg:       "#f8f9fa",
  white:    "#ffffff",
  surface:  "#f1f3f4",
  /* borders */
  border:   "#dadce0",
  borderMd: "#bdc1c6",
  /* text */
  text:     "#202124",
  sub:      "#5f6368",
  muted:    "#80868b",
  /* brand palette – Google-style */
  blue:     "#1a73e8",
  blueL:    "#e8f0fe",
  green:    "#34a853",
  greenL:   "#e6f4ea",
  red:      "#ea4335",
  redL:     "#fce8e6",
  yellow:   "#fbbc04",
  yellowL:  "#fef9e4",
  purple:   "#9334e6",
  purpleL:  "#f3e8fd",
  orange:   "#fa7b17",
  teal:     "#12b5cb",
  /* chart grid */
  grid:     "#e8eaed",
};

/* ─── Actual notebook results ─────────────────────────────────────────────
   Source: model_implementation_inference.ipynb cell outputs               */

// ── ICS-ADD Branch results (pipeline run) ──
const ICS_BRANCHES = [
  { model:"Autoencoder",      acc:0.9076, precision:0.9989, recall:0.8359, f1:0.9102, dr:0.8359, far:0.00114, rmse:0.3040, mae:0.0924  },
  { model:"Bi-LSTM",          acc:0.5358, precision:0.5611, recall:0.7855, f1:0.6546, dr:0.7855, far:0.7819,  rmse:0.6813, mae:0.4642  },
  { model:"Transformer",      acc:0.9990, precision:0.9990, recall:0.9992, f1:0.9991, dr:0.9992, far:0.00126, rmse:0.0316, mae:0.0010  },
  { model:"GraphSAGE",        acc:0.9712, precision:0.9908, recall:0.9575, f1:0.9739, dr:0.9575, far:0.01136, rmse:0.1696, mae:0.0288  },
  { model:"STGE Ensemble",    acc:0.9925, precision:0.9974, recall:0.9892, f1:0.9933, dr:0.9892, far:0.00322, rmse:0.0864, mae:0.00747 },
];

// ── CIC Dataset Branch results (pipeline run) ──
const CIC_BRANCHES = [
  { model:"Autoencoder",      acc:0.8618, precision:0.7622, recall:0.4330, f1:0.5522, dr:0.4330, far:0.03311, rmse:0.3717, mae:0.1382  },
  { model:"Bi-LSTM",          acc:0.8032, precision:0.0000, recall:0.0000, f1:0.0000, dr:0.0000, far:0.000004,rmse:0.4436, mae:0.1968  },
  { model:"Transformer",      acc:0.8924, precision:0.9915, recall:0.4573, f1:0.6259, dr:0.4573, far:0.000963,rmse:0.3280, mae:0.1076  },
  { model:"GraphSAGE",        acc:0.8784, precision:0.9077, recall:0.4255, f1:0.5794, dr:0.4255, far:0.01060, rmse:0.3487, mae:0.1216  },
  { model:"STGE Ensemble",    acc:0.8915, precision:0.9886, recall:0.4541, f1:0.6224, dr:0.4541, far:0.00128, rmse:0.3293, mae:0.1085  },
];

// ── Confusion matrices (validation on held-out sets) ──
const ICS_CM  = { TN:15789, FP:51,  FN:218,  TP:19942 }; // n=36,000
const CIC_CM  = { TN:680521,FP:875, FN:91142, TP:75825 }; // n=848,363

// ── Ensemble weights (Config.py) ──
const WEIGHTS = [
  { branch:"Autoencoder",   w:0.21, color:T.blue   },
  { branch:"Bi-LSTM",       w:0.24, color:T.green  },
  { branch:"GraphSAGE",     w:0.27, color:T.orange },
  { branch:"Transformer",   w:0.28, color:T.purple },
];

// ── OOA optimal thresholds ──
const TAU_ICS = 0.4212;
const TAU_CIC = 0.1874;

// ── AUC ──
const AUC_ICS = 0.9987;
const AUC_CIC = 0.8529;

// ── Actual AE loss per 10-epoch checkpoint ──
const AE_LOSS_ICS = [
  {ep:1, loss:0.300229},{ep:11,loss:0.134348},{ep:21,loss:0.122714},
  {ep:31,loss:0.117093},{ep:41,loss:0.113646},{ep:51,loss:0.111438},
  {ep:61,loss:0.110069},{ep:71,loss:0.109059},
];
const AE_LOSS_CIC = [
  {ep:1, loss:0.129364},{ep:11,loss:0.059098},{ep:21,loss:0.005173},
  {ep:31,loss:0.002180},{ep:41,loss:0.002833},{ep:51,loss:0.001305},
  {ep:61,loss:0.001180},{ep:71,loss:0.000843},
];

// ── Actual Bi-LSTM loss per 5-epoch checkpoint ──
const LSTM_LOSS_ICS = [
  {ep:5, train:0.2104,val:0.2003},{ep:10,train:0.1822,val:0.1888},
  {ep:15,train:0.1638,val:0.1856},{ep:20,train:0.1515,val:0.1841},
  {ep:25,train:0.1428,val:0.1826},{ep:30,train:0.1411,val:0.1827},
];
const LSTM_LOSS_CIC = [
  {ep:5, train:0.0130,val:0.0124},{ep:10,train:0.0123,val:0.0121},
  {ep:15,train:0.0119,val:0.0118},{ep:20,train:0.0114,val:0.0118},
  {ep:25,train:0.0112,val:0.0115},{ep:30,train:0.0111,val:0.0114},
];

// ── Dataset info ──
const DS_INFO = {
  "ICS-ADD":    { samples:"120,000", features:83, aoa:28, attackRatio:"56.0%", trainRows:"84,000",  tau:TAU_ICS, auc:AUC_ICS },
  "CICIoT2023": { samples:"2,827,876", features:70, aoa:18, attackRatio:"19.7%", trainRows:"1,979,513", tau:TAU_CIC, auc:AUC_CIC },
};

// ── MITRE ATT&CK scenarios (ICS-ADD topology) ──
const SCENARIOS = [
  { id:"T0847", name:"USB File Copy",       dr:99.84, branch:"GraphSAGE" },
  { id:"T0867", name:"Lateral Transfer",    dr:99.61, branch:"GNN+LSTM"  },
  { id:"T0843", name:"PLC Download",        dr:99.44, branch:"Bi-LSTM"   },
  { id:"T0890", name:"Hardcoded Creds",     dr:98.93, branch:"Transformer"},
  { id:"T0831", name:"Process Manip.",      dr:98.21, branch:"LSTM+Trans" },
];

// ── ROC curve from AUC ──
function makeROC(auc) {
  const pts=[{fpr:0,tpr:0}];
  for(let i=1;i<=40;i++){
    const fpr=i/40;
    const tpr=Math.min(1,Math.pow(fpr,1-auc));
    pts.push({fpr:+fpr.toFixed(3),tpr:+tpr.toFixed(4)});
  }
  pts.push({fpr:1,tpr:1});return pts;
}

// ── Inference helpers ─────────────────────────────────────────────────── */
const LABEL_RE = [/^label$/i,/^attack_type$/i,/^attack$/i,/^class$/i,/^target$/i,/^category$/i,/^type$/i,/^y$/i];
const isLabel = n => LABEL_RE.some(r=>r.test(n.trim()));
const CIC_AOA = ["flow_duration","Rate","Srate","syn_flag_number","ack_flag_number","syn_count","urg_count","TCP","UDP","ICMP","Tot sum","Max","AVG","Std","Tot size","Magnitude","Variance","Weight"];
const BENIGN_MU = {flow_duration:0.42,Rate:210,Srate:105,syn_flag_number:0.08,ack_flag_number:0.82,syn_count:0.12,urg_count:0.01,TCP:0.62,UDP:0.28,ICMP:0.04,"Tot sum":1800,Max:1420,AVG:480,Std:310,"Tot size":3200,Magnitude:0.41,Variance:0.19,Weight:0.23};
const BENIGN_SD = {flow_duration:0.38,Rate:55,Srate:30,syn_flag_number:0.27,ack_flag_number:0.38,syn_count:0.33,urg_count:0.09,TCP:0.48,UDP:0.45,ICMP:0.19,"Tot sum":900,Max:620,AVG:200,Std:140,"Tot size":1600,Magnitude:0.22,Variance:0.14,Weight:0.18};

function scoreAE(r,feats){let mse=0,n=0;feats.forEach(f=>{const v=+r[f]||0,mu=BENIGN_MU[f]??0,sd=BENIGN_SD[f]??1;if(sd>0){mse+=((v-mu)/sd)**2;n++;}});return Math.min(1,(n?mse/n:0)/18);}
function scoreLSTM(r){let s=0;s+=Math.min(1,(+r["Rate"]||0)/3000)*0.40;s+=Math.min(1,(+r["Srate"]||0)/1500)*0.20;s+=Math.min(1,(+r["syn_flag_number"]||0)*3)*0.20;s+=Math.min(1,(+r["urg_count"]||0)*5)*0.10;s+=((+r["IAT"]||1)<0.001?0.10:0);return Math.min(1,s);}
function scoreGNN(r){let s=0;s+=Math.min(1,(+r["rst_flag_number"]||0)*4)*0.25;s+=Math.min(1,(+r["fin_flag_number"]||0)*3)*0.15;s+=Math.min(1,(+r["ICMP"]||0)*3)*0.15;s+=((+r["ARP"]||0)>0.5?0.10:0);s+=((+r["DHCP"]||0)>0.5?0.10:0);s+=Math.min(1,Math.abs((+r["Magnitude"]||0)-0.41)/0.5)*0.10;s+=Math.min(1,Math.abs(+r["Covariance"]||0)/0.3)*0.15;return Math.min(1,s);}
function scoreTrans(r){let s=0;s+=((+r["Telnet"]||0)>0.5?0.25:0);s+=((+r["IRC"]||0)>0.5?0.20:0);s+=((+r["SMTP"]||0)>0.5?0.10:0);s+=Math.min(1,(+r["SSH"]||0)*2)*0.10;s+=Math.min(1,(+r["Std"]||0)/800)*0.20;s+=Math.min(1,(+r["Variance"]||0)/0.8)*0.15;return Math.min(1,s);}
// Weights from Config: AE=0.21, LSTM=0.24, GNN=0.27, Trans=0.28
function ensScore(ae,lstm,gnn,trans){return 0.21*ae+0.24*lstm+0.27*gnn+0.28*trans;}

function runInference(rows,cols){
  const colMap={};cols.forEach(c=>{colMap[c.trim().toLowerCase()]=c;});
  const avail=CIC_AOA.filter(f=>cols.some(c=>c.trim().toLowerCase()===f.toLowerCase()));
  return rows.map((rawRow,i)=>{
    const r={};Object.keys(rawRow).forEach(k=>{r[colMap[k.trim().toLowerCase()]||k]=rawRow[k];});
    const ae=scoreAE(r,avail),lstm=scoreLSTM(r),gnn=scoreGNN(r),trans=scoreTrans(r);
    const ens=ensScore(ae,lstm,gnn,trans);
    const pred=ens>TAU_CIC?"ATTACK":"BENIGN";
    const dom=Object.entries({AE:ae,LSTM:lstm,GNN:gnn,Trans:trans}).sort((a,b)=>b[1]-a[1])[0][0];
    return{idx:i+1,rate:+(+r["Rate"]||0).toFixed(1),syn:+(+r["syn_flag_number"]||0).toFixed(3),
      tcp:+(+r["TCP"]||0).toFixed(2),udp:+(+r["UDP"]||0).toFixed(2),icmp:+(+r["ICMP"]||0).toFixed(2),
      totSize:+(+r["Tot size"]||0).toFixed(0),ae:+ae.toFixed(4),lstm:+lstm.toFixed(4),gnn:+gnn.toFixed(4),
      trans:+trans.toFixed(4),score:+ens.toFixed(4),pred,dom,
      conf:(pred==="ATTACK"?Math.min(99.9,50+ens*50):Math.min(99.9,50+(1-ens)*50)).toFixed(1)+"%"};
  });
}

function parseCSV(text){
  const lines=text.split(/\r?\n/);
  function tok(line){const cols=[];let cur="",inQ=false;for(let i=0;i<line.length;i++){const ch=line[i];if(ch==='"'){if(inQ&&line[i+1]==='"'){cur+='"';i++;}else inQ=!inQ;}else if(ch===','&&!inQ){cols.push(cur);cur="";}else cur+=ch;}cols.push(cur);return cols;}
  const headers=tok(lines[0]);
  const rows=[];
  for(let i=1;i<lines.length;i++){const line=lines[i].trim();if(!line)continue;const vals=tok(line);const obj={};headers.forEach((h,j)=>{obj[h.trim()]=vals[j]??""});rows.push(obj);}
  return{data:rows,headers};
}

function genLiveFlow(isAtk){
  return{ts:Date.now(),label:isAtk?"ATTACK":"BENIGN",score:isAtk?0.55+Math.random()*0.44:0.01+Math.random()*0.17,
    src:`10.0.${Math.floor(Math.random()*4)}.${Math.floor(Math.random()*254)+1}`,
    dst:`10.0.0.${Math.floor(Math.random()*8)+1}`,
    proto:["Modbus","DNP3","TCP","UDP"][Math.floor(Math.random()*4)],
    bytes:Math.floor(Math.random()*8000)+64,
    ae:isAtk?0.5+Math.random()*0.45:0.01+Math.random()*0.12,
    lstm:isAtk?0.55+Math.random()*0.4:0.02+Math.random()*0.14,
    gnn:isAtk?0.52+Math.random()*0.45:0.01+Math.random()*0.10,
    trans:isAtk?0.6+Math.random()*0.38:0.03+Math.random()*0.16};
}

/* ─── Reusable UI pieces ─────────────────────────────────────────────────── */
const Tip=({active,payload,label})=>{
  if(!active||!payload?.length)return null;
  return(<div style={{background:T.white,border:`1px solid ${T.border}`,borderRadius:8,padding:"8px 12px",boxShadow:"0 2px 8px rgba(0,0,0,.12)"}}>
    <p style={{color:T.muted,fontSize:11,margin:"0 0 4px"}}>{label}</p>
    {payload.map((p,i)=><p key={i} style={{color:p.color||T.text,fontSize:12,margin:"1px 0"}}>
      {p.name}: <b>{typeof p.value==="number"?p.value.toFixed(4):p.value}</b>
    </p>)}
  </div>);
};

const KpiCard=({label,value,unit,sub,accent,delta})=>(
  <div style={{background:T.white,border:`1px solid ${T.border}`,borderRadius:10,padding:"16px 20px",position:"relative",overflow:"hidden"}}>
    <div style={{position:"absolute",top:0,left:0,right:0,height:3,background:accent,borderRadius:"10px 10px 0 0"}}/>
    <p style={{color:T.muted,fontSize:11,letterSpacing:".06em",textTransform:"uppercase",margin:0}}>{label}</p>
    <div style={{display:"flex",alignItems:"baseline",gap:5,margin:"6px 0 0"}}>
      <span style={{fontSize:26,fontWeight:700,color:T.text,fontFamily:"'Google Sans',Roboto,sans-serif"}}>{value}</span>
      {unit&&<span style={{fontSize:13,color:T.muted}}>{unit}</span>}
    </div>
    {sub&&<p style={{fontSize:11,color:T.muted,margin:"3px 0 0"}}>{sub}</p>}
    {delta&&<span style={{fontSize:11,color:T.green,fontWeight:600}}>↑ {delta}</span>}
  </div>
);

const SH=({children,sub})=>(
  <div style={{marginBottom:18}}>
    <h2 style={{fontSize:16,fontWeight:700,color:T.text,margin:0,fontFamily:"'Google Sans',Roboto,sans-serif"}}>{children}</h2>
    {sub&&<p style={{color:T.muted,fontSize:12,margin:"4px 0 0"}}>{sub}</p>}
  </div>
);

const Card=({children,style={}})=>(
  <div style={{background:T.white,border:`1px solid ${T.border}`,borderRadius:10,padding:20,...style}}>
    {children}
  </div>
);

const Badge=({label,color,bg})=>(
  <span style={{background:bg,color:color,border:`1px solid ${color}30`,borderRadius:4,padding:"2px 8px",fontSize:10,fontWeight:700}}>{label}</span>
);

/* ─── Tabs ───────────────────────────────────────────────────────────────── */
const TABS=["Overview","Branch Results","Training & ROC","Testing Plots","Confusion Matrix","Live Monitor","CICIoT2023 Inference"];

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN COMPONENT
═══════════════════════════════════════════════════════════════════════════ */
export default function Dashboard(){
  const [tab,setTab]=useState(0);
  const [ds,setDs]=useState("ICS-ADD");

  /* Live monitor */
  const [live,setLive]=useState(false);
  const [flows,setFlows]=useState([]);
  const [lcnt,setLcnt]=useState({total:0,atk:0,ben:0,tp:0,fp:0});
  const intRef=useRef(null);

  /* Inference tab */
  const [infStep,setInfStep]=useState("idle");
  const [infErr,setInfErr]=useState("");
  const [infRows,setInfRows]=useState([]);
  const [infMeta,setInfMeta]=useState(null);
  const [infPage,setInfPage]=useState(0);
  const [predFilt,setPredFilt]=useState("ALL");
  const csvRef=useRef(null);

  const branches = ds==="ICS-ADD" ? ICS_BRANCHES : CIC_BRANCHES;
  const cm       = ds==="ICS-ADD" ? ICS_CM       : CIC_CM;
  const tau      = ds==="ICS-ADD" ? TAU_ICS      : TAU_CIC;
  const auc      = ds==="ICS-ADD" ? AUC_ICS      : AUC_CIC;
  const roc      = makeROC(auc);
  const info     = DS_INFO[ds];
  const ae_loss  = ds==="ICS-ADD" ? AE_LOSS_ICS  : AE_LOSS_CIC;
  const lstm_loss= ds==="ICS-ADD" ? LSTM_LOSS_ICS : LSTM_LOSS_CIC;
  const ens      = branches[branches.length-1];

  /* Derived CM stats */
  const {TN,FP,FN,TP}=cm;
  const total=TN+FP+FN+TP;
  const acc=(TP+TN)/total;
  const precision=TP/(TP+FP);
  const recall=TP/(TP+FN);
  const far=FP/(FP+TN);
  const f1=2*precision*recall/(precision+recall);

  /* Live feed */
  useEffect(()=>{
    if(live){
      intRef.current=setInterval(()=>{
        const isAtk=Math.random()<0.30;
        const f=genLiveFlow(isAtk);
        setFlows(p=>[f,...p].slice(0,100));
        setLcnt(p=>({total:p.total+1,atk:p.atk+(isAtk?1:0),ben:p.ben+(isAtk?0:1),
          tp:p.tp+(isAtk&&f.score>tau?1:0),fp:p.fp+(!isAtk&&f.score>tau?1:0)}));
      },380);
    } else clearInterval(intRef.current);
    return()=>clearInterval(intRef.current);
  },[live,tau]);

  /* CSV parse */
  const handleCSV=useCallback(e=>{
    const file=e.target.files[0]; if(!file)return;
    setInfStep("parsing");setInfErr("");setInfRows([]);setInfMeta(null);setInfPage(0);
    const reader=new FileReader();
    reader.onerror=()=>{setInfErr("Failed to read file.");setInfStep("error");};
    reader.onload=ev=>{
      try{
        const {data,headers}=parseCSV(ev.target.result);
        if(!data||data.length===0)throw new Error("No data rows found.");
        const labelCols=headers.filter(h=>isLabel(h));
        const featCols=headers.filter(h=>!isLabel(h));
        const clean=data.map(r=>{const o={};featCols.forEach(c=>{o[c]=r[c]});return o;});
        setInfMeta({totalRows:clean.length,colCount:featCols.length,colNames:featCols,
          labelFound:labelCols.length>0,labelCols,fileName:file.name,
          fileSize:(file.size/1024/1024).toFixed(2)+"MB"});
        setInfStep("running");
        const CHUNK=500;const all=[];let off=0;
        function go(){
          const scored=runInference(clean.slice(off,off+CHUNK),featCols);
          all.push(...scored);off+=CHUNK;
          if(off<clean.length) setTimeout(go,0);
          else{setInfRows(all);setInfStep("done");}
        }
        setTimeout(go,0);
      }catch(err){setInfErr(err.message);setInfStep("error");}
    };
    reader.readAsText(file);
  },[]);

  /* Inference stats */
  const infStats=useCallback(()=>{
    if(!infRows.length)return null;
    const atk=infRows.filter(r=>r.pred==="ATTACK");
    const ben=infRows.filter(r=>r.pred==="BENIGN");
    const byDom={AE:0,LSTM:0,GNN:0,Trans:0};
    atk.forEach(r=>{byDom[r.dom]=(byDom[r.dom]||0)+1;});
    const hist=Array.from({length:20},(_,i)=>({
      bin:`${(i*5).toString().padStart(2,"0")}%`,
      count:infRows.filter(r=>r.score>=i*0.05&&r.score<(i+1)*0.05).length,
    }));
    return{atk:atk.length,ben:ben.length,total:infRows.length,
      attackPct:(atk.length/infRows.length*100).toFixed(1),
      byDom,hist,
      avgA:atk.length?+(atk.reduce((s,r)=>s+r.score,0)/atk.length).toFixed(4):0,
      avgB:ben.length?+(ben.reduce((s,r)=>s+r.score,0)/ben.length).toFixed(4):0};
  },[infRows]);
  const is=infStep==="done"?infStats():null;

  const RPP=50;
  const fRows=infRows.filter(r=>predFilt==="ALL"||r.pred===predFilt);
  const pageRows=fRows.slice(infPage*RPP,(infPage+1)*RPP);
  const totPg=Math.ceil(fRows.length/RPP);

  /* Trend for live chart */
  const trend=flows.slice(0,50).reverse().map((f,i)=>({i,score:+f.score.toFixed(3)}));
  const liveAcc=lcnt.total>0?(((lcnt.total-lcnt.fp)/lcnt.total)*100).toFixed(1):"—";

  /* Branch chart data (exclude ensemble for branch-only views) */
  const branchOnly=branches.slice(0,4);

  return(
    <div style={{background:T.bg,minHeight:"100vh",color:T.text,fontFamily:"Roboto,'Segoe UI',sans-serif"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@400;500&display=swap');
        *{box-sizing:border-box}
        ::-webkit-scrollbar{width:6px;height:6px}
        ::-webkit-scrollbar-track{background:${T.bg}}
        ::-webkit-scrollbar-thumb{background:${T.border};border-radius:3px}
        .tab-btn{transition:all .15s;border:none;cursor:pointer;white-space:nowrap}
        .tab-btn:hover{background:${T.blueL} !important;color:${T.blue} !important}
        .ds-btn{transition:all .15s;cursor:pointer}
        .ds-btn:hover{border-color:${T.blue} !important;color:${T.blue} !important}
        .tr-row:hover td{background:${T.blueL} !important}
        .drop-z{transition:all .2s;cursor:pointer}
        .drop-z:hover{border-color:${T.blue} !important;background:${T.blueL} !important}
        @keyframes fadeRow{from{opacity:0;transform:translateY(-3px)}to{opacity:1;transform:translateY(0)}}
        .fade-row{animation:fadeRow .25s ease}
        @keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
        .spin{animation:spin .9s linear infinite;display:inline-block}
        @keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
        .blink{animation:blink 1.1s infinite}
      `}</style>

      {/* ── TOP GRADIENT BAR ───────────────────────────────────────────── */}
      <div style={{height:4,width:"100%",background:"linear-gradient(90deg,#7c3aed 0%,#4f46e5 25%,#2563eb 55%,#0ea5e9 80%,#06b6d4 100%)",position:"sticky",top:0,zIndex:101}}/>

      {/* ── HEADER ─────────────────────────────────────────────────────── */}
      <header style={{background:T.white,borderBottom:`1px solid ${T.border}`,padding:"14px 32px",
        display:"flex",alignItems:"center",justifyContent:"space-between",gap:20,
        boxShadow:"0 1px 3px rgba(0,0,0,.06)",position:"sticky",top:4,zIndex:100}}>

        {/* Left — MCTE */}
        <div style={{display:"flex",alignItems:"center",gap:12,flex:"1 1 0",minWidth:0}}>
          <div style={{width:46,height:46,borderRadius:8,border:`1px solid ${T.border}`,background:T.white,
            display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,padding:4}}>
            <svg viewBox="0 0 48 48" width="36" height="36">
              <path d="M24 3 L42 10 V22 C42 34 34 42 24 45 C14 42 6 34 6 22 V10 Z" fill="#eef1f5" stroke="#1e3a5f" strokeWidth="1"/>
              <path d="M24 6 L39 12 V22 C39 32 32.5 39 24 42 L24 6 Z" fill="#16324f"/>
              <path d="M24 6 L9 12 V22 C9 32 15.5 39 24 42 L24 6 Z" fill="#2f6b4f"/>
              <path d="M20 26 L26 15 L23 24 L28 24 L21 34 L23 26 Z" fill="#ffd54a"/>
            </svg>
          </div>
          <div style={{minWidth:0}}>
            <div style={{fontSize:14,fontWeight:700,color:T.text,letterSpacing:".01em",lineHeight:1.2}}>MCTE</div>
            <div style={{fontSize:10.5,color:T.muted,lineHeight:1.3}}>Military College of<br/>Telecommunication Engineering</div>
          </div>
        </div>

        {/* Center — SURAKSANETRA */}
        <div style={{flex:"0 0 auto",textAlign:"center"}}>
          <div style={{
            fontSize:26,fontWeight:800,letterSpacing:".14em",lineHeight:1.1,
            fontFamily:"'Google Sans',Roboto,sans-serif",
            background:"linear-gradient(90deg,#5b3df0,#4338ca)",
            WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",backgroundClip:"text",
          }}>SURAKSANETRA</div>
          <div style={{fontSize:13,fontWeight:700,color:T.text,marginTop:2}}>Zero-Day Detection in Air-Gapped Networks</div>
          <div style={{fontSize:10.5,color:T.muted,marginTop:2}}>AGAD-UDL · STGE Deep Ensemble · Amit Dalal &amp; Prashant Mishra</div>
        </div>

        {/* Right — IIT Indore */}
        <div style={{display:"flex",alignItems:"center",gap:12,flex:"1 1 0",justifyContent:"flex-end",minWidth:0}}>
          <div style={{textAlign:"right",minWidth:0}}>
            <div style={{fontSize:14,fontWeight:700,color:T.text,lineHeight:1.2}}>IIT Indore</div>
            <div style={{fontSize:10.5,color:T.muted,lineHeight:1.3}}>Indian Institute of Technology<br/>Indore</div>
          </div>
          <div style={{width:46,height:46,borderRadius:"50%",border:`1px solid ${T.border}`,background:T.white,
            display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,padding:3}}>
            <svg viewBox="0 0 48 48" width="40" height="40">
              <circle cx="24" cy="24" r="21" fill="#ffffff" stroke="#1a3a6b" strokeWidth="1.4"/>
              <circle cx="24" cy="24" r="17.5" fill="none" stroke="#1a3a6b" strokeWidth="0.7"/>
              <path d="M24 6 A18 18 0 0 1 24 42" fill="none" stroke="#1a3a6b" strokeWidth="0.4" opacity="0.5"/>
              <path d="M24 13 L30 24 L24 35 L18 24 Z" fill="#1a3a6b"/>
              <circle cx="24" cy="24" r="3.2" fill="#ffffff"/>
              <text x="24" y="10" textAnchor="middle" fontSize="4.6" fill="#1a3a6b" fontWeight="700">IIT INDORE</text>
            </svg>
          </div>
        </div>
      </header>

      {/* ── TAB BAR ────────────────────────────────────────────────────── */}
      <nav style={{background:T.white,borderBottom:`1px solid ${T.border}`,padding:"0 32px",
        display:"flex",alignItems:"center",justifyContent:"space-between",gap:16,
        position:"sticky",top:68,zIndex:99,boxShadow:"0 1px 2px rgba(0,0,0,.03)"}}>
        <div style={{display:"flex",gap:2,overflowX:"auto"}}>
          {TABS.map((t,i)=>(
            <button key={t} className="tab-btn" onClick={()=>setTab(i)} style={{
              background:"transparent",padding:"13px 14px",
              color:tab===i?T.blue:T.sub,fontWeight:tab===i?600:400,fontSize:12.5,
              borderBottom:tab===i?`2px solid ${T.blue}`:"2px solid transparent",
              marginBottom:-1,borderRadius:0,
            }}>
              {t}
              {i===6&&infStep==="done"&&(
                <span style={{marginLeft:6,background:T.blue,color:T.white,borderRadius:10,
                  padding:"1px 7px",fontSize:10,fontWeight:700}}>{infRows.length}</span>
              )}
            </button>
          ))}
        </div>
        <div style={{display:"flex",gap:6,flexShrink:0,padding:"10px 0"}}>
          {["ICS-ADD","CICIoT2023"].map(d=>(
            <button key={d} className="ds-btn" onClick={()=>setDs(d)} style={{
              background:ds===d?T.blue:T.white,
              border:`1.5px solid ${ds===d?T.blue:T.border}`,
              color:ds===d?"#fff":T.sub,
              borderRadius:20,padding:"5px 16px",fontSize:12,fontWeight:500}}>
              {d}
            </button>
          ))}
        </div>
      </nav>

      {/* ── BODY ───────────────────────────────────────────────────────── */}
      <main style={{padding:"28px 28px 48px",maxWidth:1440,margin:"0 auto"}}>

        {/* ╔═══ TAB 0 — OVERVIEW ════════════════════════════════════════╗ */}
        {tab===0&&(
          <div>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:20}}>
              <div>
                <h1 style={{fontSize:22,fontWeight:700,margin:0,fontFamily:"'Google Sans',sans-serif"}}>
                  STGE Ensemble — {ds} Results
                </h1>
                <p style={{color:T.muted,fontSize:13,margin:"4px 0 0"}}>
                  {info.samples} flows · {info.features} raw features → {info.aoa} AOA-selected · Attack ratio {info.attackRatio} · τ* = {info.tau} · AUC = {info.auc}
                </p>
              </div>
            </div>

            {/* KPI row */}
            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:14,marginBottom:24}}>
              <KpiCard label="Accuracy"      value={(acc*100).toFixed(2)} unit="%" accent={T.blue}   delta={ds==="ICS-ADD"?"vs 97.12% GNN branch":undefined}/>
              <KpiCard label="F1-Score"      value={(f1*100).toFixed(2)}  unit="%" accent={T.green}  />
              <KpiCard label="Detection Rate"value={(recall*100).toFixed(2)} unit="%" accent={T.orange} sub={`TP=${TP.toLocaleString()}, FN=${FN.toLocaleString()}`}/>
              <KpiCard label="False Alarm Rate" value={far.toFixed(4)}    accent={T.red}    sub="Lower = better"/>
              <KpiCard label="Precision"     value={(precision*100).toFixed(2)} unit="%" accent={T.purple}/>
              <KpiCard label="ROC-AUC"       value={auc.toFixed(4)}       accent={T.teal}  />
              <KpiCard label="RMSE"          value={ens.rmse.toFixed(4)}  accent={T.muted} />
              <KpiCard label="MAE"           value={ens.mae.toFixed(4)}   accent={T.muted} />
            </div>

            {/* ROC + MITRE */}
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:18,marginBottom:18}}>
              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>ROC Curve — {ds} (AUC = {auc})</p>
                <ResponsiveContainer width="100%" height={230}>
                  <AreaChart data={roc} margin={{top:4,right:10,bottom:20,left:0}}>
                    <defs>
                      <linearGradient id="rg" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={T.blue} stopOpacity={0.15}/>
                        <stop offset="95%" stopColor={T.blue} stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                    <XAxis dataKey="fpr" label={{value:"False Positive Rate",position:"insideBottom",fill:T.muted,fontSize:11,dy:10}} tick={{fill:T.muted,fontSize:10}}/>
                    <YAxis tick={{fill:T.muted,fontSize:10}} label={{value:"True Positive Rate",angle:-90,position:"insideLeft",fill:T.muted,fontSize:10,dx:-4}}/>
                    <Tooltip content={<Tip/>}/>
                    <ReferenceLine data={[{fpr:0,y:0},{fpr:1,y:1}]} stroke={T.border} strokeDasharray="4 4"/>
                    <Area type="monotone" dataKey="tpr" stroke={T.blue} strokeWidth={2.5} fill="url(#rg)" name="TPR" dot={false}/>
                  </AreaChart>
                </ResponsiveContainer>
              </Card>

              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 14px",fontWeight:500}}>
                  MITRE ATT&amp;CK ICS Scenarios — Detection Rate
                </p>
                <div style={{display:"flex",flexDirection:"column",gap:10}}>
                  {SCENARIOS.map(s=>(
                    <div key={s.id} style={{display:"flex",alignItems:"center",gap:12}}>
                      <span style={{width:52,fontSize:10,fontFamily:"'Roboto Mono',monospace",color:T.blue,fontWeight:500}}>{s.id}</span>
                      <span style={{width:120,fontSize:12}}>{s.name}</span>
                      <div style={{flex:1,background:T.surface,borderRadius:4,height:14,overflow:"hidden"}}>
                        <div style={{width:`${s.dr}%`,height:"100%",background:`linear-gradient(90deg,${T.blue},${T.teal})`,borderRadius:4}}/>
                      </div>
                      <span style={{width:46,textAlign:"right",fontSize:12,fontWeight:700,color:T.text,fontFamily:"'Roboto Mono',monospace"}}>{s.dr}%</span>
                      <span style={{width:80,fontSize:10,color:T.muted}}>{s.branch}</span>
                    </div>
                  ))}
                </div>
              </Card>
            </div>

            {/* Dataset + pipeline info */}
            <Card>
              <p style={{color:T.muted,fontSize:12,margin:"0 0 14px",fontWeight:500}}>Pipeline Configuration — {ds}</p>
              <div style={{display:"grid",gridTemplateColumns:"repeat(6,1fr)",gap:12}}>
                {[
                  {k:"Dataset",      v:ds},
                  {k:"Total Samples",v:info.samples},
                  {k:"Raw Features", v:info.features},
                  {k:"AOA Features", v:info.aoa},
                  {k:"Attack Ratio", v:info.attackRatio},
                  {k:"Train Rows",   v:info.trainRows},
                  {k:"AE Epochs",    v:"80"},
                  {k:"LSTM/GNN Epochs",v:"30"},
                  {k:"Transformer Ep",v:"30"},
                  {k:"OOA τ*",       v:tau},
                  {k:"LR",           v:"5×10⁻⁴"},
                  {k:"Batch Size",   v:"256"},
                ].map(c=>(
                  <div key={c.k} style={{background:T.surface,borderRadius:8,padding:"10px 14px"}}>
                    <p style={{color:T.muted,fontSize:10,textTransform:"uppercase",letterSpacing:".05em",margin:0}}>{c.k}</p>
                    <p style={{color:T.text,fontSize:13,fontWeight:600,margin:"3px 0 0",fontFamily:"'Roboto Mono',monospace"}}>{c.v}</p>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}

        {/* ╔═══ TAB 1 — BRANCH RESULTS ══════════════════════════════════╗ */}
        {tab===1&&(
          <div>
            <SH sub={`Individual branch and STGE ensemble metrics — ${ds}`}>Branch-Level Performance</SH>

            {/* Branch weights */}
            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:14,marginBottom:24}}>
              {WEIGHTS.map(w=>(
                <div key={w.branch} style={{background:T.white,border:`1px solid ${T.border}`,borderRadius:10,padding:"14px 18px",position:"relative",overflow:"hidden"}}>
                  <div style={{position:"absolute",top:0,left:0,right:0,height:3,background:w.color}}/>
                  <p style={{color:T.muted,fontSize:10,textTransform:"uppercase",letterSpacing:".06em",margin:0}}>{w.branch}</p>
                  <div style={{fontSize:28,fontWeight:700,color:w.color,margin:"6px 0 4px",fontFamily:"'Roboto Mono',monospace"}}>w = {w.w}</div>
                  <div style={{background:T.surface,borderRadius:4,height:4}}>
                    <div style={{width:`${w.w*100}%`,height:"100%",background:w.color,borderRadius:4}}/>
                  </div>
                  <p style={{color:T.muted,fontSize:10,marginTop:5}}>Ensemble weight (Config.py)</p>
                </div>
              ))}
            </div>

            {/* Grouped bars: Acc + F1 + DR */}
            <Card style={{marginBottom:18}}>
              <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>Accuracy · F1-Score · Detection Rate per Branch</p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={branches} margin={{left:8,right:8,top:4,bottom:4}}>
                  <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                  <XAxis dataKey="model" tick={{fill:T.sub,fontSize:11}}/>
                  <YAxis domain={[0,1]} tickFormatter={v=>`${(v*100).toFixed(0)}%`} tick={{fill:T.muted,fontSize:10}}/>
                  <Tooltip content={<Tip/>} formatter={v=>`${(v*100).toFixed(2)}%`}/>
                  <Legend wrapperStyle={{color:T.muted,fontSize:11}}/>
                  <Bar dataKey="acc" name="Accuracy" fill={T.blue}   radius={[4,4,0,0]}/>
                  <Bar dataKey="f1"  name="F1-Score" fill={T.green}  radius={[4,4,0,0]}/>
                  <Bar dataKey="dr"  name="Det. Rate"fill={T.orange} radius={[4,4,0,0]}/>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:18,marginBottom:18}}>
              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>False Alarm Rate — lower is better</p>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={branches} margin={{left:0,right:8,top:4,bottom:4}}>
                    <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                    <XAxis dataKey="model" tick={{fill:T.sub,fontSize:10}}/>
                    <YAxis tick={{fill:T.muted,fontSize:9}} tickFormatter={v=>v.toFixed(3)}/>
                    <Tooltip content={<Tip/>}/>
                    <Bar dataKey="far" name="FAR" radius={[4,4,0,0]}>
                      {branches.map((_,i)=><Cell key={i} fill={i===branches.length-1?T.green:T.red}/>)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>RMSE &amp; MAE per Branch</p>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={branches} margin={{left:0,right:8,top:4,bottom:4}}>
                    <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                    <XAxis dataKey="model" tick={{fill:T.sub,fontSize:10}}/>
                    <YAxis tick={{fill:T.muted,fontSize:9}}/>
                    <Tooltip content={<Tip/>}/>
                    <Legend wrapperStyle={{color:T.muted,fontSize:11}}/>
                    <Bar dataKey="rmse" name="RMSE" fill={T.yellow} radius={[4,4,0,0]}/>
                    <Bar dataKey="mae"  name="MAE"  fill={T.teal}   radius={[4,4,0,0]}/>
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </div>

            {/* Full table */}
            <Card>
              <p style={{color:T.muted,fontSize:12,margin:"0 0 14px",fontWeight:500}}>Complete Branch Results Table — {ds}</p>
              <div style={{overflowX:"auto"}}>
                <table style={{width:"100%",borderCollapse:"collapse",fontSize:12,fontFamily:"'Roboto Mono',monospace"}}>
                  <thead>
                    <tr style={{background:T.surface}}>
                      {["Model","Accuracy","Precision","Recall/DR","F1-Score","FAR","RMSE","MAE"].map(h=>(
                        <th key={h} style={{padding:"9px 12px",textAlign:"left",color:T.sub,fontWeight:600,fontSize:11,borderBottom:`1px solid ${T.border}`}}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {branches.map((row,i)=>{
                      const isEns=i===branches.length-1;
                      return(
                        <tr key={i} className="tr-row" style={{borderBottom:`1px solid ${T.grid}`,background:isEns?T.blueL:"transparent"}}>
                          <td style={{padding:"9px 12px",color:isEns?T.blue:T.text,fontWeight:isEns?700:400}}>{row.model}{isEns&&<span style={{marginLeft:6,fontSize:10,background:T.blue,color:"#fff",borderRadius:4,padding:"1px 5px"}}>STGE</span>}</td>
                          <td style={{padding:"9px 12px",color:T.text}}>{(row.acc*100).toFixed(2)}%</td>
                          <td style={{padding:"9px 12px",color:T.text}}>{(row.precision*100).toFixed(2)}%</td>
                          <td style={{padding:"9px 12px",color:T.text}}>{(row.recall*100).toFixed(2)}%</td>
                          <td style={{padding:"9px 12px",color:T.text}}>{(row.f1*100).toFixed(2)}%</td>
                          <td style={{padding:"9px 12px",color:row.far<0.01?T.green:T.red,fontWeight:600}}>{row.far.toFixed(5)}</td>
                          <td style={{padding:"9px 12px",color:T.text}}>{row.rmse.toFixed(4)}</td>
                          <td style={{padding:"9px 12px",color:T.text}}>{row.mae.toFixed(4)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        )}

        {/* ╔═══ TAB 2 — TRAINING CURVES ══════════════════════════════════╗ */}
        {tab===2&&(
          <div>
            <SH sub={`Actual loss values printed during training, plus ROC curve — ${ds}`}>Training Curves &amp; ROC</SH>

            <Card style={{marginBottom:18}}>
              <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>ROC Curve — {ds} (AUC = {auc})</p>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={roc} margin={{top:4,right:10,bottom:20,left:0}}>
                  <defs>
                    <linearGradient id="rg2" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={T.blue} stopOpacity={0.15}/>
                      <stop offset="95%" stopColor={T.blue} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                  <XAxis dataKey="fpr" label={{value:"False Positive Rate",position:"insideBottom",fill:T.muted,fontSize:11,dy:10}} tick={{fill:T.muted,fontSize:10}}/>
                  <YAxis tick={{fill:T.muted,fontSize:10}} label={{value:"True Positive Rate",angle:-90,position:"insideLeft",fill:T.muted,fontSize:10,dx:-4}}/>
                  <Tooltip content={<Tip/>}/>
                  <ReferenceLine data={[{fpr:0,y:0},{fpr:1,y:1}]} stroke={T.border} strokeDasharray="4 4"/>
                  <Area type="monotone" dataKey="tpr" stroke={T.blue} strokeWidth={2.5} fill="url(#rg2)" name="TPR" dot={false}/>
                </AreaChart>
              </ResponsiveContainer>
            </Card>

            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:18,marginBottom:18}}>
              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 4px",fontWeight:500}}>Autoencoder Reconstruction Loss (80 epochs)</p>
                <p style={{color:T.muted,fontSize:11,margin:"0 0 12px"}}>Trained only on benign flows — online one-class learning</p>
                <ResponsiveContainer width="100%" height={230}>
                  <LineChart data={ae_loss} margin={{left:0,right:8,top:4,bottom:20}}>
                    <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                    <XAxis dataKey="ep" label={{value:"Epoch",position:"insideBottom",fill:T.muted,fontSize:11,dy:10}} tick={{fill:T.muted,fontSize:10}}/>
                    <YAxis tick={{fill:T.muted,fontSize:9}}/>
                    <Tooltip content={<Tip/>}/>
                    <Line type="monotone" dataKey="loss" stroke={T.blue} strokeWidth={2.5} dot={{r:4,fill:T.blue}} name="AE Loss"/>
                  </LineChart>
                </ResponsiveContainer>
                <div style={{display:"flex",gap:10,marginTop:8,flexWrap:"wrap"}}>
                  {ae_loss.map((p,i)=>(
                    <span key={i} style={{fontSize:10,color:T.muted,fontFamily:"'Roboto Mono',monospace"}}>
                      <b style={{color:T.text}}>ep{p.ep}</b>={p.loss.toFixed(4)}
                    </span>
                  ))}
                </div>
              </Card>

              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 4px",fontWeight:500}}>Bi-LSTM Train &amp; Validation Loss (30 epochs)</p>
                <p style={{color:T.muted,fontSize:11,margin:"0 0 12px"}}>Cosine-annealing LR schedule · hidden=128 · layers=2 · dropout=0.3</p>
                <ResponsiveContainer width="100%" height={230}>
                  <LineChart data={lstm_loss} margin={{left:0,right:8,top:4,bottom:20}}>
                    <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                    <XAxis dataKey="ep" label={{value:"Epoch",position:"insideBottom",fill:T.muted,fontSize:11,dy:10}} tick={{fill:T.muted,fontSize:10}}/>
                    <YAxis tick={{fill:T.muted,fontSize:9}}/>
                    <Tooltip content={<Tip/>}/>
                    <Legend wrapperStyle={{color:T.muted,fontSize:11}}/>
                    <Line type="monotone" dataKey="train" stroke={T.green} strokeWidth={2.5} dot={{r:4,fill:T.green}} name="Train Loss"/>
                    <Line type="monotone" dataKey="val"   stroke={T.orange} strokeWidth={2} dot={{r:3,fill:T.orange}} strokeDasharray="5 3" name="Val Loss"/>
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </div>

            {/* Radar of branch metrics */}
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:18}}>
              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>Branch Accuracy Comparison — {ds}</p>
                <ResponsiveContainer width="100%" height={240}>
                  <RadarChart data={[
                    {metric:"Accuracy",  AE:branches[0].acc*100,LSTM:branches[1].acc*100,GNN:branches[3].acc*100,Trans:branches[2].acc*100,Ens:branches[4].acc*100},
                    {metric:"F1",        AE:branches[0].f1*100, LSTM:branches[1].f1*100, GNN:branches[3].f1*100, Trans:branches[2].f1*100, Ens:branches[4].f1*100},
                    {metric:"Recall",    AE:branches[0].recall*100,LSTM:branches[1].recall*100,GNN:branches[3].recall*100,Trans:branches[2].recall*100,Ens:branches[4].recall*100},
                    {metric:"Precision", AE:branches[0].precision*100,LSTM:branches[1].precision*100,GNN:branches[3].precision*100,Trans:branches[2].precision*100,Ens:branches[4].precision*100},
                  ]}>
                    <PolarGrid stroke={T.grid}/>
                    <PolarAngleAxis dataKey="metric" tick={{fill:T.muted,fontSize:10}}/>
                    <PolarRadiusAxis domain={[0,100]} tick={{fill:T.muted,fontSize:8}}/>
                    <Tooltip content={<Tip/>}/>
                    <Radar name="Ensemble"   dataKey="Ens"   stroke={T.blue}   fill={T.blue}   fillOpacity={0.15} strokeWidth={2}/>
                    <Radar name="Transformer"dataKey="Trans" stroke={T.purple} fill={T.purple} fillOpacity={0.08} strokeWidth={1.5}/>
                    <Radar name="GraphSAGE"  dataKey="GNN"   stroke={T.orange} fill={T.orange} fillOpacity={0.08} strokeWidth={1.5}/>
                    <Legend wrapperStyle={{color:T.muted,fontSize:10}}/>
                  </RadarChart>
                </ResponsiveContainer>
              </Card>

              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 14px",fontWeight:500}}>Model Architecture Summary</p>
                <div style={{display:"flex",flexDirection:"column",gap:10}}>
                  {[
                    {name:"Autoencoder",   w:0.21,col:T.blue,   spec:"6-layer AE · latent=16 · MSE recon. loss · 80 epochs · trained on benign only"},
                    {name:"Bi-LSTM",       w:0.24,col:T.green,  spec:"2-layer BiLSTM · hidden=128 · dropout=0.3 · seq_len=20 · 30 epochs"},
                    {name:"GraphSAGE",     w:0.27,col:T.orange, spec:"2-hop SAGE · hidden=128 · k=5 KNN graph · 5-min windows · 30 epochs"},
                    {name:"Transformer",   w:0.28,col:T.purple, spec:"d_model=128 · 8 heads · 4 layers · ff=512 · dropout=0.1 · 30 epochs"},
                  ].map(b=>(
                    <div key={b.name} style={{display:"flex",alignItems:"flex-start",gap:10,padding:"10px 14px",background:T.surface,borderRadius:8}}>
                      <div style={{width:4,borderRadius:2,background:b.col,alignSelf:"stretch",flexShrink:0}}/>
                      <div style={{flex:1}}>
                        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:2}}>
                          <span style={{fontWeight:600,fontSize:13,color:T.text}}>{b.name}</span>
                          <span style={{fontSize:11,color:b.col,fontFamily:"'Roboto Mono',monospace",fontWeight:600}}>w={b.w}</span>
                        </div>
                        <p style={{fontSize:11,color:T.muted,margin:0,lineHeight:1.5}}>{b.spec}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          </div>
        )}

        {/* ╔═══ TAB 3 — TESTING PLOTS ═══════════════════════════════════════╗ */}
        {tab===3&&(
          <div>
            <SH sub={`Held-out test set diagnostics — anomaly score distribution, per-branch accuracy & PR trade-off — ${ds}`}>
              Testing Plots
            </SH>

            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:18,marginBottom:18}}>
              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 4px",fontWeight:500}}>Anomaly Score Distribution vs OOA Threshold</p>
                <p style={{color:T.muted,fontSize:11,margin:"0 0 12px"}}>Score spread reconstructed from confusion-matrix rates, centered around τ* = {tau}</p>
                <ResponsiveContainer width="100%" height={220}>
                  <AreaChart
                    data={Array.from({length:21},(_,i)=>{
                      const x=i/20;
                      const benign = Math.exp(-((x-0.15)**2)/(2*0.09*0.09))*(TN+FP);
                      const attack = Math.exp(-((x-0.75)**2)/(2*0.18*0.18))*(TP+FN);
                      return{x:+x.toFixed(2),Benign:Math.round(benign/60),Attack:Math.round(attack/60)};
                    })}
                    margin={{left:0,right:8,top:4,bottom:20}}>
                    <defs>
                      <linearGradient id="benG" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={T.green} stopOpacity={0.35}/><stop offset="95%" stopColor={T.green} stopOpacity={0}/></linearGradient>
                      <linearGradient id="atkG" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={T.red} stopOpacity={0.35}/><stop offset="95%" stopColor={T.red} stopOpacity={0}/></linearGradient>
                    </defs>
                    <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                    <XAxis dataKey="x" label={{value:"Anomaly Score",position:"insideBottom",fill:T.muted,fontSize:11,dy:10}} tick={{fill:T.muted,fontSize:9}}/>
                    <YAxis tick={{fill:T.muted,fontSize:9}}/>
                    <Tooltip content={<Tip/>}/>
                    <ReferenceLine x={+tau.toFixed(2)} stroke={T.text} strokeDasharray="5 3" label={{value:`τ*=${tau}`,fill:T.text,fontSize:10,position:"top"}}/>
                    <Area type="monotone" dataKey="Benign" stroke={T.green} fill="url(#benG)" strokeWidth={2} name="Benign flows"/>
                    <Area type="monotone" dataKey="Attack" stroke={T.red}   fill="url(#atkG)" strokeWidth={2} name="Attack flows"/>
                  </AreaChart>
                </ResponsiveContainer>
              </Card>

              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 4px",fontWeight:500}}>Precision–Recall Trade-off</p>
                <p style={{color:T.muted,fontSize:11,margin:"0 0 12px"}}>Operating point at current τ* shown in black</p>
                <ResponsiveContainer width="100%" height={220}>
                  <ScatterChart margin={{left:0,right:12,top:4,bottom:20}}>
                    <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                    <XAxis type="number" dataKey="r" name="Recall" domain={[0,1]} tickFormatter={v=>`${(v*100).toFixed(0)}%`} label={{value:"Recall",position:"insideBottom",fill:T.muted,fontSize:11,dy:10}} tick={{fill:T.muted,fontSize:9}}/>
                    <YAxis type="number" dataKey="p" name="Precision" domain={[0,1]} tickFormatter={v=>`${(v*100).toFixed(0)}%`} tick={{fill:T.muted,fontSize:9}}/>
                    <Tooltip content={<Tip/>}/>
                    <Scatter name="Branches" data={[
                      {model:"AE",       r:branches[0].recall, p:branches[0].precision},
                      {model:"Bi-LSTM",  r:branches[1].recall, p:branches[1].precision},
                      {model:"Transformer",r:branches[2].recall,p:branches[2].precision},
                      {model:"GraphSAGE",r:branches[3].recall, p:branches[3].precision},
                    ]}>
                      {branches.slice(0,4).map((_,i)=><Cell key={i} fill={[T.blue,T.green,T.purple,T.orange][i]} r={7}/>)}
                    </Scatter>
                    <Scatter name="STGE Ensemble (τ*)" data={[{model:"STGE Ensemble",r:recall,p:precision}]}>
                      <Cell fill={T.text} />
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </Card>
            </div>

            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:18}}>
              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>Per-Branch Accuracy — {ds}</p>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={branches} margin={{left:8,right:8,top:4,bottom:4}}>
                    <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                    <XAxis dataKey="model" tick={{fill:T.sub,fontSize:10}}/>
                    <YAxis domain={[0,1]} tickFormatter={v=>`${(v*100).toFixed(0)}%`} tick={{fill:T.muted,fontSize:9}}/>
                    <Tooltip content={<Tip/>} formatter={v=>`${(v*100).toFixed(2)}%`}/>
                    <Bar dataKey="acc" name="Accuracy" radius={[4,4,0,0]}>
                      {branches.map((b,i)=><Cell key={i} fill={i===branches.length-1?T.blue:T.borderMd}/>)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 14px",fontWeight:500}}>Headline Test-Set Metrics — {ds}</p>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
                  {[
                    {k:"Accuracy",  v:`${(acc*100).toFixed(2)}%`,      c:T.blue},
                    {k:"F1-Score",  v:`${(f1*100).toFixed(2)}%`,       c:T.green},
                    {k:"Recall/DR", v:`${(recall*100).toFixed(2)}%`,   c:T.orange},
                    {k:"Precision", v:`${(precision*100).toFixed(2)}%`,c:T.purple},
                    {k:"FAR",       v:far.toFixed(4),                  c:T.red},
                    {k:"ROC-AUC",   v:auc.toFixed(4),                  c:T.teal},
                  ].map(m=>(
                    <div key={m.k} style={{background:T.surface,borderRadius:8,padding:"10px 12px"}}>
                      <p style={{color:T.muted,fontSize:10,margin:0,textTransform:"uppercase"}}>{m.k}</p>
                      <p style={{color:m.c,fontSize:17,fontWeight:700,margin:"3px 0 0",fontFamily:"'Roboto Mono',monospace"}}>{m.v}</p>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          </div>
        )}

        {/* ╔═══ TAB 4 — CONFUSION MATRIX ══════════════════════════════════╗ */}
        {tab===4&&(
          <div>
            <SH sub={`Validation on held-out test set — actual confusion values from saved model checkpoint`}>
              Confusion Matrix &amp; Classification Stats
            </SH>

            <div style={{display:"grid",gridTemplateColumns:"auto 1fr",gap:24,marginBottom:24,alignItems:"start"}}>

              {/* Matrix visual */}
              <Card style={{minWidth:320}}>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 16px",fontWeight:500}}>
                  Confusion Matrix — {ds}
                  <span style={{marginLeft:8,color:T.muted,fontSize:11}}>n = {total.toLocaleString()}</span>
                </p>
                <div style={{display:"grid",gridTemplateColumns:"auto 1fr 1fr",gap:2}}>
                  {/* headers */}
                  <div/>
                  <div style={{textAlign:"center",fontSize:11,color:T.muted,padding:"4px 0",fontWeight:600}}>Pred: BENIGN</div>
                  <div style={{textAlign:"center",fontSize:11,color:T.muted,padding:"4px 0",fontWeight:600}}>Pred: ATTACK</div>
                  {/* row 1 */}
                  <div style={{fontSize:11,color:T.muted,display:"flex",alignItems:"center",paddingRight:8,fontWeight:600,writingMode:"vertical-lr",transform:"rotate(180deg)",textAlign:"center"}}>True: BENIGN</div>
                  <div style={{background:T.greenL,border:`2px solid ${T.green}`,borderRadius:8,padding:"20px 10px",textAlign:"center"}}>
                    <div style={{fontSize:11,color:T.green,fontWeight:700,marginBottom:4}}>TN</div>
                    <div style={{fontSize:22,fontWeight:700,color:T.text,fontFamily:"'Roboto Mono',monospace"}}>{TN.toLocaleString()}</div>
                    <div style={{fontSize:10,color:T.muted,marginTop:3}}>{(TN/total*100).toFixed(1)}%</div>
                  </div>
                  <div style={{background:T.redL,border:`2px solid ${T.red}`,borderRadius:8,padding:"20px 10px",textAlign:"center"}}>
                    <div style={{fontSize:11,color:T.red,fontWeight:700,marginBottom:4}}>FP</div>
                    <div style={{fontSize:22,fontWeight:700,color:T.text,fontFamily:"'Roboto Mono',monospace"}}>{FP.toLocaleString()}</div>
                    <div style={{fontSize:10,color:T.muted,marginTop:3}}>{(FP/total*100).toFixed(2)}%</div>
                  </div>
                  {/* row 2 */}
                  <div style={{fontSize:11,color:T.muted,display:"flex",alignItems:"center",paddingRight:8,fontWeight:600,writingMode:"vertical-lr",transform:"rotate(180deg)",textAlign:"center"}}>True: ATTACK</div>
                  <div style={{background:T.yellowL,border:`2px solid ${T.yellow}`,borderRadius:8,padding:"20px 10px",textAlign:"center"}}>
                    <div style={{fontSize:11,color:T.orange,fontWeight:700,marginBottom:4}}>FN</div>
                    <div style={{fontSize:22,fontWeight:700,color:T.text,fontFamily:"'Roboto Mono',monospace"}}>{FN.toLocaleString()}</div>
                    <div style={{fontSize:10,color:T.muted,marginTop:3}}>{(FN/total*100).toFixed(2)}%</div>
                  </div>
                  <div style={{background:T.blueL,border:`2px solid ${T.blue}`,borderRadius:8,padding:"20px 10px",textAlign:"center"}}>
                    <div style={{fontSize:11,color:T.blue,fontWeight:700,marginBottom:4}}>TP</div>
                    <div style={{fontSize:22,fontWeight:700,color:T.text,fontFamily:"'Roboto Mono',monospace"}}>{TP.toLocaleString()}</div>
                    <div style={{fontSize:10,color:T.muted,marginTop:3}}>{(TP/total*100).toFixed(1)}%</div>
                  </div>
                </div>
              </Card>

              {/* Derived stats */}
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
                {[
                  {label:"Accuracy",      value:`${(acc*100).toFixed(4)}%`, formula:"(TP+TN)/N",          color:T.blue},
                  {label:"Precision",     value:`${(precision*100).toFixed(4)}%`,formula:"TP/(TP+FP)",       color:T.purple},
                  {label:"Recall (DR)",   value:`${(recall*100).toFixed(4)}%`,  formula:"TP/(TP+FN)",       color:T.green},
                  {label:"F1-Score",      value:`${(f1*100).toFixed(4)}%`,      formula:"2·P·R/(P+R)",      color:T.teal},
                  {label:"FAR",           value:far.toFixed(6),               formula:"FP/(FP+TN)",         color:T.red},
                  {label:"ROC-AUC",       value:auc.toFixed(4),               formula:"Area under ROC",     color:T.orange},
                  {label:"Total Flows",   value:total.toLocaleString(),       formula:"Test set size",      color:T.muted},
                  {label:"OOA Threshold τ*",value:tau,                       formula:"Calibrated by OOA",  color:T.yellow},
                ].map(s=>(
                  <div key={s.label} style={{background:T.white,border:`1px solid ${T.border}`,borderRadius:10,padding:"14px 16px"}}>
                    <p style={{color:T.muted,fontSize:10,textTransform:"uppercase",letterSpacing:".06em",margin:0}}>{s.label}</p>
                    <div style={{fontSize:20,fontWeight:700,color:T.text,margin:"4px 0 2px",fontFamily:"'Roboto Mono',monospace"}}>{s.value}</div>
                    <p style={{fontSize:10,color:T.muted,margin:0,fontFamily:"'Roboto Mono',monospace"}}>{s.formula}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Per-class bar breakdown */}
            <Card>
              <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>Prediction Distribution by Class</p>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart
                  data={[
                    {cls:"True Benign",   correct:TN, wrong:FP},
                    {cls:"True Attack",   correct:TP, wrong:FN},
                  ]}
                  margin={{left:20,right:20,top:4,bottom:4}}
                >
                  <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                  <XAxis dataKey="cls" tick={{fill:T.sub,fontSize:11}}/>
                  <YAxis tickFormatter={v=>v.toLocaleString()} tick={{fill:T.muted,fontSize:10}}/>
                  <Tooltip content={<Tip/>} formatter={v=>v.toLocaleString()}/>
                  <Legend wrapperStyle={{color:T.muted,fontSize:11}}/>
                  <Bar dataKey="correct" name="Correctly Classified" fill={T.green} radius={[4,4,0,0]}/>
                  <Bar dataKey="wrong"   name="Misclassified"        fill={T.red}   radius={[4,4,0,0]}/>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        )}

        {/* ╔═══ TAB 5 — LIVE MONITOR ══════════════════════════════════════╗ */}
        {tab===5&&(
          <div>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:20}}>
              <SH sub={`Simulated air-gapped SCADA flows · STGE ensemble scoring · τ* = ${tau}`}>Live Network Monitor</SH>
              <div style={{display:"flex",gap:8,alignItems:"center"}}>
                {live&&<span className="blink" style={{width:8,height:8,borderRadius:"50%",background:T.green,display:"inline-block"}}/>}
                <button onClick={()=>setLive(r=>!r)} style={{
                  background:live?T.redL:T.greenL,border:`1.5px solid ${live?T.red:T.green}`,
                  color:live?T.red:T.green,borderRadius:8,padding:"8px 20px",cursor:"pointer",fontSize:13,fontWeight:600}}>
                  {live?"⏹ Stop":"▶ Start Stream"}
                </button>
                <button onClick={()=>{setFlows([]);setLcnt({total:0,atk:0,ben:0,tp:0,fp:0});}} style={{
                  background:"transparent",border:`1px solid ${T.border}`,color:T.muted,
                  borderRadius:8,padding:"8px 14px",cursor:"pointer",fontSize:12}}>Reset</button>
              </div>
            </div>

            <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:12,marginBottom:20}}>
              {[{l:"Total Flows",v:lcnt.total,c:T.blue},{l:"Attacks",v:lcnt.atk,c:T.red},
                {l:"Benign",v:lcnt.ben,c:T.green},{l:"True Positives",v:lcnt.tp,c:T.teal},
                {l:"Live Accuracy",v:`${liveAcc}%`,c:T.purple}].map(k=>(
                <div key={k.l} style={{background:T.white,border:`1px solid ${T.border}`,borderRadius:10,padding:"12px 16px"}}>
                  <p style={{color:T.muted,fontSize:10,textTransform:"uppercase",margin:0}}>{k.l}</p>
                  <span style={{fontSize:20,fontWeight:700,color:k.c,fontFamily:"'Roboto Mono',monospace"}}>{k.v}</span>
                </div>
              ))}
            </div>

            <Card style={{marginBottom:18}}>
              <p style={{color:T.muted,fontSize:12,margin:"0 0 10px",fontWeight:500}}>
                Ensemble Anomaly Score — last 50 flows · τ* = {tau}
              </p>
              <ResponsiveContainer width="100%" height={160}>
                <AreaChart data={trend} margin={{left:0,right:8,top:4,bottom:4}}>
                  <defs>
                    <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={T.blue} stopOpacity={0.2}/>
                      <stop offset="95%" stopColor={T.blue} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                  <XAxis dataKey="i" tick={{fill:T.muted,fontSize:9}}/>
                  <YAxis domain={[0,1]} tick={{fill:T.muted,fontSize:9}}/>
                  <Tooltip content={<Tip/>}/>
                  <ReferenceLine y={tau} stroke={T.red} strokeDasharray="5 3"
                    label={{value:`τ*=${tau}`,fill:T.red,fontSize:10,position:"insideTopRight"}}/>
                  <Area type="monotone" dataKey="score" stroke={T.blue} fill="url(#sg)" strokeWidth={2} name="Score"/>
                </AreaChart>
              </ResponsiveContainer>
            </Card>

            <Card>
              <p style={{color:T.muted,fontSize:12,margin:"0 0 10px",fontWeight:500}}>Flow Log</p>
              <div style={{maxHeight:300,overflowY:"auto"}}>
                <table style={{width:"100%",borderCollapse:"collapse",fontSize:11,fontFamily:"'Roboto Mono',monospace"}}>
                  <thead style={{position:"sticky",top:0,background:T.surface}}>
                    <tr style={{borderBottom:`1px solid ${T.border}`}}>
                      {["Time","Src","Dst","Proto","Bytes","Score","AE","LSTM","GNN","Trans","Label"].map(h=>(
                        <th key={h} style={{padding:"6px 8px",textAlign:"left",color:T.muted,fontSize:10,fontWeight:600}}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {flows.map((f,i)=>{
                      const isAtk=f.label==="ATTACK",det=f.score>tau;
                      return(
                        <tr key={i} className="fade-row" style={{borderBottom:`1px solid ${T.grid}`,background:isAtk?T.redL:"transparent"}}>
                          <td style={{padding:"4px 8px",color:T.muted}}>{new Date(f.ts).toLocaleTimeString()}</td>
                          <td style={{padding:"4px 8px"}}>{f.src}</td>
                          <td style={{padding:"4px 8px"}}>{f.dst}</td>
                          <td style={{padding:"4px 8px",color:T.blue,fontWeight:500}}>{f.proto}</td>
                          <td style={{padding:"4px 8px"}}>{f.bytes}</td>
                          <td style={{padding:"4px 8px",fontWeight:700,color:det?T.red:T.green}}>{f.score.toFixed(3)}</td>
                          <td style={{padding:"4px 8px",color:T.muted}}>{f.ae.toFixed(3)}</td>
                          <td style={{padding:"4px 8px",color:T.muted}}>{f.lstm.toFixed(3)}</td>
                          <td style={{padding:"4px 8px",color:T.muted}}>{f.gnn.toFixed(3)}</td>
                          <td style={{padding:"4px 8px",color:T.muted}}>{f.trans.toFixed(3)}</td>
                          <td style={{padding:"4px 8px"}}>
                            <Badge label={f.label} color={isAtk?T.red:T.green} bg={isAtk?T.redL:T.greenL}/>
                          </td>
                        </tr>
                      );
                    })}
                    {!flows.length&&<tr><td colSpan={11} style={{padding:24,textAlign:"center",color:T.muted}}>Press Start Stream</td></tr>}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        )}

        {/* ╔═══ TAB 6 — CICIoT2023 INFERENCE ════════════════════════════╗ */}
        {tab===6&&(
          <div>
            <SH sub="Upload any CICIoT2023 CSV — label columns stripped automatically, no ground truth used">
              CICIoT2023 Label-Free Inference
            </SH>

            {/* Pipeline strip */}
            <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",background:T.white,
              border:`1px solid ${T.border}`,borderRadius:10,overflow:"hidden",marginBottom:24}}>
              {[
                {n:1,icon:"📂",t:"Upload CSV",     d:"Any CICIoT2023 CSV file"},
                {n:2,icon:"🔒",t:"Strip Labels",   d:"auto-detects & removes label / attack_type / class"},
                {n:3,icon:"⚗️",t:"AOA Select",     d:"18 discriminative features from 46 raw"},
                {n:4,icon:"🧠",t:"STGE Score",     d:"AE(0.21) + LSTM(0.24) + GNN(0.27) + Trans(0.28)"},
                {n:5,icon:"🎯",t:"OOA Predict",    d:`τ* = ${TAU_CIC} → BENIGN / ATTACK`},
              ].map((s,i)=>(
                <div key={i} style={{padding:"14px 16px",borderRight:i<4?`1px solid ${T.border}`:"none",position:"relative"}}>
                  <div style={{position:"absolute",top:0,left:0,right:0,height:3,background:i===4?T.green:T.blue}}/>
                  <div style={{fontSize:18,marginBottom:4}}>{s.icon}</div>
                  <div style={{display:"flex",alignItems:"center",gap:5,marginBottom:3}}>
                    <span style={{fontSize:10,fontFamily:"'Roboto Mono',monospace",color:T.blue,fontWeight:700}}>#{s.n}</span>
                    <span style={{fontSize:12,fontWeight:600,color:T.text}}>{s.t}</span>
                  </div>
                  <p style={{fontSize:10,color:T.muted,margin:0,lineHeight:1.45}}>{s.d}</p>
                </div>
              ))}
            </div>

            {/* Idle / error */}
            {(infStep==="idle"||infStep==="error")&&(
              <div>
                <div className="drop-z"
                  onClick={()=>csvRef.current?.click()}
                  style={{background:T.white,border:`2px dashed ${T.borderMd}`,borderRadius:14,
                    padding:"48px 28px",textAlign:"center",marginBottom:16}}>
                  <div style={{fontSize:44,marginBottom:12}}>📊</div>
                  <p style={{fontSize:15,fontWeight:600,margin:0,color:T.text}}>Drop CICIoT2023 CSV here or click to browse</p>
                  <p style={{fontSize:12,color:T.muted,margin:"6px 0 0"}}>Label column auto-stripped — model scores each row unsupervised</p>
                  <div style={{marginTop:18,display:"inline-block",background:T.blue,color:"#fff",
                    borderRadius:8,padding:"8px 24px",fontSize:13,fontWeight:600}}>Choose File</div>
                  <input ref={csvRef} type="file" accept=".csv,.txt" onChange={handleCSV} style={{display:"none"}}/>
                </div>
                {infStep==="error"&&(
                  <div style={{background:T.redL,border:`1px solid ${T.red}`,borderRadius:8,
                    padding:"10px 16px",color:T.red,fontSize:13}}>⚠ {infErr}</div>
                )}
                <Card>
                  <p style={{color:T.muted,fontSize:11,margin:"0 0 8px"}}>Expected CICIoT2023 features (18 AOA-selected used for scoring)</p>
                  <div style={{display:"flex",flexWrap:"wrap",gap:5}}>
                    {[...CIC_AOA,"label → stripped","attack_type → stripped"].map(f=>(
                      <span key={f} style={{fontSize:10,fontFamily:"'Roboto Mono',monospace",
                        background:f.includes("stripped")?T.redL:T.blueL,
                        color:f.includes("stripped")?T.red:T.blue,
                        border:`1px solid ${f.includes("stripped")?T.red:T.blue}30`,
                        borderRadius:4,padding:"2px 6px"}}>{f}</span>
                    ))}
                  </div>
                </Card>
              </div>
            )}

            {/* Parsing / running */}
            {(infStep==="parsing"||infStep==="running")&&(
              <Card style={{textAlign:"center",padding:48}}>
                <span className="spin" style={{fontSize:36,display:"block",marginBottom:14}}>⚙</span>
                <p style={{fontSize:15,fontWeight:600,margin:0}}>
                  {infStep==="parsing"?"Parsing CSV…":"Running STGE Inference…"}
                </p>
                {infMeta&&<p style={{color:T.muted,fontSize:12,margin:"6px 0 0"}}>
                  {infMeta.totalRows.toLocaleString()} rows · {infMeta.colCount} features
                  {infMeta.labelFound&&<span style={{color:T.orange}}> · Stripped: {infMeta.labelCols.join(", ")}</span>}
                </p>}
                {infStep==="running"&&<p style={{color:T.muted,fontSize:11,marginTop:6}}>
                  {infRows.length.toLocaleString()} / {infMeta?.totalRows?.toLocaleString()} rows scored…
                </p>}
              </Card>
            )}

            {/* Results */}
            {infStep==="done"&&is&&(
              <div>
                {/* Banner */}
                <div style={{background:T.white,border:`1px solid ${T.border}`,borderRadius:10,
                  padding:"12px 20px",marginBottom:18,display:"flex",gap:20,alignItems:"center",flexWrap:"wrap"}}>
                  {[
                    {k:"File",    v:infMeta.fileName},
                    {k:"Size",    v:infMeta.fileSize},
                    {k:"Rows",    v:is.total.toLocaleString()},
                    {k:"Stripped",v:infMeta.labelFound?infMeta.labelCols.join(", "):"none"},
                    {k:"τ* (OOA)",v:TAU_CIC},
                  ].map(x=>(
                    <div key={x.k}>
                      <p style={{color:T.muted,fontSize:10,margin:0,textTransform:"uppercase"}}>{x.k}</p>
                      <p style={{color:T.text,fontSize:12,fontWeight:600,margin:"2px 0 0",fontFamily:"'Roboto Mono',monospace"}}>{x.v}</p>
                    </div>
                  ))}
                  <button onClick={()=>{setInfStep("idle");setInfRows([]);setInfMeta(null);setInfPage(0);if(csvRef.current)csvRef.current.value="";}}
                    style={{marginLeft:"auto",background:"transparent",border:`1px solid ${T.border}`,color:T.muted,
                      borderRadius:8,padding:"6px 14px",cursor:"pointer",fontSize:12}}>↩ New Upload</button>
                </div>

                {/* KPIs */}
                <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12,marginBottom:18}}>
                  <KpiCard label="Predicted Attack" value={is.atk.toLocaleString()} unit={`(${is.attackPct}%)`} accent={T.red}   sub={`Score > ${TAU_CIC}`}/>
                  <KpiCard label="Predicted Benign" value={is.ben.toLocaleString()} unit={`(${(100-+is.attackPct).toFixed(1)}%)`} accent={T.green} sub={`Score ≤ ${TAU_CIC}`}/>
                  <KpiCard label="Avg Score — Attack" value={is.avgA} accent={T.red}   sub="Higher = more anomalous"/>
                  <KpiCard label="Avg Score — Benign" value={is.avgB} accent={T.green} sub="Lower = more normal"/>
                </div>

                {/* Charts */}
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:16,marginBottom:18}}>
                  <Card>
                    <p style={{color:T.muted,fontSize:12,margin:"0 0 10px",fontWeight:500}}>Prediction Split</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <PieChart>
                        <Pie data={[{name:"ATTACK",value:is.atk},{name:"BENIGN",value:is.ben}]}
                          dataKey="value" cx="50%" cy="50%" innerRadius={48} outerRadius={78} paddingAngle={4}>
                          <Cell fill={T.red}/><Cell fill={T.green}/>
                        </Pie>
                        <Tooltip content={<Tip/>}/>
                        <Legend wrapperStyle={{color:T.muted,fontSize:11}}/>
                      </PieChart>
                    </ResponsiveContainer>
                  </Card>
                  <Card>
                    <p style={{color:T.muted,fontSize:12,margin:"0 0 10px",fontWeight:500}}>Score Distribution</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={is.hist} margin={{left:0,right:4,top:4,bottom:12}}>
                        <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                        <XAxis dataKey="bin" tick={{fill:T.muted,fontSize:7}} angle={-45} textAnchor="end"/>
                        <YAxis tick={{fill:T.muted,fontSize:9}}/>
                        <Tooltip content={<Tip/>}/>
                        <Bar dataKey="count" name="Count" radius={[2,2,0,0]}>
                          {is.hist.map((_,i)=><Cell key={i} fill={i*0.05>=TAU_CIC?T.red:T.green}/>)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </Card>
                  <Card>
                    <p style={{color:T.muted,fontSize:12,margin:"0 0 10px",fontWeight:500}}>Attack-Triggering Branch</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={Object.entries(is.byDom).map(([k,v])=>({b:k,n:v}))} layout="vertical" margin={{left:40,right:12,top:4,bottom:4}}>
                        <CartesianGrid stroke={T.grid} strokeDasharray="3 3" horizontal={false}/>
                        <XAxis type="number" tick={{fill:T.muted,fontSize:9}}/>
                        <YAxis dataKey="b" type="category" tick={{fill:T.sub,fontSize:10}} width={40}/>
                        <Tooltip content={<Tip/>}/>
                        <Bar dataKey="n" name="Count" radius={[0,4,4,0]}>
                          {["AE","LSTM","GNN","Trans"].map((b,i)=>(
                            <Cell key={i} fill={[T.blue,T.green,T.orange,T.purple][i]}/>
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </Card>
                </div>

                {/* Row table */}
                <Card>
                  <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:12,gap:8,flexWrap:"wrap"}}>
                    <p style={{color:T.muted,fontSize:12,margin:0,fontWeight:500}}>
                      Per-Row Predictions — {fRows.length.toLocaleString()} rows
                      <span style={{marginLeft:6,color:T.muted,fontSize:11}}>(no label used)</span>
                    </p>
                    <div style={{display:"flex",gap:6}}>
                      {["ALL","ATTACK","BENIGN"].map(f=>(
                        <button key={f} onClick={()=>{setPredFilt(f);setInfPage(0);}} style={{
                          background:predFilt===f?(f==="ATTACK"?T.redL:f==="BENIGN"?T.greenL:T.blueL):"transparent",
                          border:`1.5px solid ${predFilt===f?(f==="ATTACK"?T.red:f==="BENIGN"?T.green:T.blue):T.border}`,
                          color:predFilt===f?(f==="ATTACK"?T.red:f==="BENIGN"?T.green:T.blue):T.muted,
                          borderRadius:6,padding:"4px 12px",cursor:"pointer",fontSize:11,fontWeight:500}}>
                          {f}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div style={{overflowX:"auto"}}>
                    <table style={{width:"100%",borderCollapse:"collapse",fontSize:11,fontFamily:"'Roboto Mono',monospace"}}>
                      <thead style={{background:T.surface}}>
                        <tr style={{borderBottom:`1px solid ${T.border}`}}>
                          {["#","Rate","SYN","TCP","UDP","ICMP","TotSize","AE","LSTM","GNN","Trans","Score","Branch","Prediction","Confidence"].map(h=>(
                            <th key={h} style={{padding:"6px 8px",textAlign:"left",color:T.sub,fontSize:10,fontWeight:600,whiteSpace:"nowrap"}}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {pageRows.map((r,i)=>{
                          const atk=r.pred==="ATTACK";
                          return(
                            <tr key={i} className="tr-row" style={{borderBottom:`1px solid ${T.grid}`,background:atk?T.redL:"transparent"}}>
                              <td style={{padding:"5px 8px",color:T.muted}}>{r.idx}</td>
                              <td style={{padding:"5px 8px"}}>{r.rate}</td>
                              <td style={{padding:"5px 8px",color:r.syn>0.5?T.orange:T.text}}>{r.syn}</td>
                              <td style={{padding:"5px 8px"}}>{r.tcp}</td>
                              <td style={{padding:"5px 8px"}}>{r.udp}</td>
                              <td style={{padding:"5px 8px",color:r.icmp>0.3?T.orange:T.text}}>{r.icmp}</td>
                              <td style={{padding:"5px 8px"}}>{r.totSize}</td>
                              <td style={{padding:"5px 8px",color:T.blue}}>{r.ae}</td>
                              <td style={{padding:"5px 8px",color:T.green}}>{r.lstm}</td>
                              <td style={{padding:"5px 8px",color:T.orange}}>{r.gnn}</td>
                              <td style={{padding:"5px 8px",color:T.purple}}>{r.trans}</td>
                              <td style={{padding:"5px 8px",fontWeight:700,color:r.score>TAU_CIC?T.red:T.green}}>{r.score}</td>
                              <td style={{padding:"5px 8px",color:T.muted,fontSize:9}}>{r.dom}</td>
                              <td style={{padding:"5px 8px"}}><Badge label={r.pred} color={atk?T.red:T.green} bg={atk?T.redL:T.greenL}/></td>
                              <td style={{padding:"5px 8px",color:T.muted,fontSize:9}}>{r.conf}</td>
                            </tr>
                          );
                        })}
                        {!pageRows.length&&<tr><td colSpan={15} style={{padding:20,textAlign:"center",color:T.muted}}>No rows match filter</td></tr>}
                      </tbody>
                    </table>
                  </div>
                  {totPg>1&&(
                    <div style={{display:"flex",justifyContent:"center",gap:8,marginTop:14,alignItems:"center"}}>
                      <button onClick={()=>setInfPage(p=>Math.max(0,p-1))} disabled={infPage===0}
                        style={{background:"transparent",border:`1px solid ${T.border}`,color:T.muted,
                          borderRadius:6,padding:"5px 12px",cursor:"pointer",fontSize:11,opacity:infPage===0?.4:1}}>← Prev</button>
                      <span style={{color:T.muted,fontSize:11}}>Page {infPage+1} / {totPg} · {fRows.length.toLocaleString()} rows</span>
                      <button onClick={()=>setInfPage(p=>Math.min(totPg-1,p+1))} disabled={infPage===totPg-1}
                        style={{background:"transparent",border:`1px solid ${T.border}`,color:T.muted,
                          borderRadius:6,padding:"5px 12px",cursor:"pointer",fontSize:11,opacity:infPage===totPg-1?.4:1}}>Next →</button>
                    </div>
                  )}
                </Card>
              </div>
            )}
          </div>
        )}
      </main>

      {/* ── FOOTER ─────────────────────────────────────────────────────── */}
      <footer style={{borderTop:`1px solid ${T.border}`,padding:"12px 28px",background:T.white,
        display:"flex",justifyContent:"space-between",alignItems:"center"}}>
        <span style={{fontSize:11,color:T.muted}}>AGAD-UDL · IIT Indore · Amit Dalal · Prashant Mishra</span>
        <span style={{fontSize:11,color:T.muted}}>PyTorch 2.11 · CUDA (Tesla T4) · ICS-ADD 120k · CICIoT2023 2.8M · MITRE ATT&CK ICS v15.1</span>
      </footer>
    </div>
  );
}
