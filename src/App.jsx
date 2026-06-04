/**
 * AGAD-UDL  —  Interactive Performance Dashboard
 * IIT Indore  |  Amit Dalal, Prashant Mishra
 * All metrics sourced directly from model_implementation_inference.ipynb
 * Plots reconstructed from notebook outputs:
 *   - ROC Curves (combined, all models / both datasets)
 *   - Training Loss Curves (all models / both datasets)
 *   - Testing Plots ICS  (Confusion / ROC / PR / Score-Dist / Per-Branch / Metrics)
 *   - Testing Plots CIC  (same 6-panel layout)
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
  bg:       "#f8f9fa",
  white:    "#ffffff",
  surface:  "#f1f3f4",
  border:   "#dadce0",
  borderMd: "#bdc1c6",
  text:     "#202124",
  sub:      "#5f6368",
  muted:    "#80868b",
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
  brown:    "#795548",
  pink:     "#e91e63",
  gray:     "#607d8b",
  olive:    "#afb42b",
  grid:     "#e8eaed",
};

/* ═══════════════════════════════════════════════════════════════════════════
   ACTUAL NOTEBOOK RESULTS  (model_implementation_inference.ipynb cell outputs)
═══════════════════════════════════════════════════════════════════════════ */

// ── ICS-ADD Branch results (pipeline run) ──
const ICS_BRANCHES = [
  { model:"Autoencoder",   acc:0.9076, precision:0.9989, recall:0.8359, f1:0.9102, dr:0.8359, far:0.00114, rmse:0.3040, mae:0.0924  },
  { model:"Bi-LSTM",       acc:0.5358, precision:0.5611, recall:0.7855, f1:0.6546, dr:0.7855, far:0.7819,  rmse:0.6813, mae:0.4642  },
  { model:"Transformer",   acc:0.9990, precision:0.9990, recall:0.9992, f1:0.9991, dr:0.9992, far:0.00126, rmse:0.0316, mae:0.0010  },
  { model:"GraphSAGE",     acc:0.9712, precision:0.9908, recall:0.9575, f1:0.9739, dr:0.9575, far:0.01136, rmse:0.1696, mae:0.0288  },
  { model:"STGE Ensemble", acc:0.9925, precision:0.9974, recall:0.9892, f1:0.9933, dr:0.9892, far:0.00322, rmse:0.0864, mae:0.00747 },
];

// ── CIC Dataset Branch results (pipeline run) ──
const CIC_BRANCHES = [
  { model:"Autoencoder",   acc:0.8618, precision:0.7622, recall:0.4330, f1:0.5522, dr:0.4330, far:0.03311, rmse:0.3717, mae:0.1382  },
  { model:"Bi-LSTM",       acc:0.8032, precision:0.0000, recall:0.0000, f1:0.0000, dr:0.0000, far:0.000004,rmse:0.4436, mae:0.1968  },
  { model:"Transformer",   acc:0.8924, precision:0.9915, recall:0.4573, f1:0.6259, dr:0.4573, far:0.000963,rmse:0.3280, mae:0.1076  },
  { model:"GraphSAGE",     acc:0.8784, precision:0.9077, recall:0.4255, f1:0.5794, dr:0.4255, far:0.01060, rmse:0.3487, mae:0.1216  },
  { model:"STGE Ensemble", acc:0.8915, precision:0.9886, recall:0.4541, f1:0.6224, dr:0.4541, far:0.00128, rmse:0.3293, mae:0.1085  },
];

// ── Confusion matrices (validation on held-out sets) ──
const ICS_CM = { TN:15789, FP:51,  FN:218,  TP:19942 };  // n=36,000
const CIC_CM = { TN:680521,FP:875, FN:91142, TP:75825 }; // n=848,363

// ── Ensemble weights (Config.py) ──
const WEIGHTS = [
  { branch:"Autoencoder", w:0.21, color:T.blue   },
  { branch:"Bi-LSTM",     w:0.24, color:T.green  },
  { branch:"GraphSAGE",   w:0.27, color:T.orange },
  { branch:"Transformer", w:0.28, color:T.purple },
];

// ── OOA optimal thresholds ──
const TAU_ICS = 0.4212;
const TAU_CIC = 0.1874;

// ── AUC (ensemble) ──
const AUC_ICS = 0.9987;
const AUC_CIC = 0.8529;

/* ── Combined ROC AUCs  (from "ROC Curves" notebook figure) ──────────────── */
const ROC_AUCS = {
  "ICS-ADD": [
    { name:"Autoencoder",   auc:0.9943, color:T.blue   },
    { name:"Bi-LSTM",       auc:0.5021, color:T.orange },
    { name:"Transformer",   auc:0.9999, color:T.green  },
    { name:"GraphSAGE",     auc:0.9943, color:T.red    },
    { name:"STGE Ensemble", auc:0.9987, color:T.purple },
  ],
  "CICIoT2023": [
    { name:"Autoencoder",   auc:0.7889, color:T.brown },
    { name:"Bi-LSTM",       auc:0.4989, color:T.pink  },
    { name:"Transformer",   auc:0.9075, color:T.gray  },
    { name:"GraphSAGE",     auc:0.8034, color:T.olive },
    { name:"STGE Ensemble", auc:0.8511, color:T.teal  },
  ],
};

/* ── Per-dataset TESTING-PLOT data  (from "AGAD-UDL — Testing Plots" figures) ─ */
const TEST_PLOTS = {
  "ICS-ADD": {
    cm:  { TN:15789, FP:51, FN:218, TP:19942 },
    auc: { Ensemble:0.999, AE:0.994, LSTM:0.502, Transformer:1.000, GraphSAGE:0.994 },
    ap:  0.999,
    tau: 0.421,
    branchAcc: [
      { b:"AE", v:0.545 }, { b:"LSTM", v:0.536 }, { b:"Transformer", v:0.999 },
      { b:"GraphSAGE", v:0.971 }, { b:"Ensemble", v:0.993 },
    ],
    metrics: [
      { m:"Accuracy", v:0.993 }, { m:"Precision", v:0.997 }, { m:"Recall", v:0.989 },
      { m:"F1", v:0.993 }, { m:"FAR", v:0.003 },
    ],
    hist: { bMu:0.33, bSd:0.075, bPk:14.2, aMu:0.75, aSd:0.13, aPk:5.4 },
  },
  "CICIoT2023": {
    cm:  { TN:680521, FP:875, FN:91142, TP:75825 },
    auc: { Ensemble:0.853, AE:0.788, LSTM:0.500, Transformer:0.907, GraphSAGE:0.804 },
    ap:  0.683,
    tau: 0.187,
    branchAcc: [
      { b:"AE", v:0.803 }, { b:"LSTM", v:0.803 }, { b:"Transformer", v:0.892 },
      { b:"GraphSAGE", v:0.878 }, { b:"Ensemble", v:0.892 },
    ],
    metrics: [
      { m:"Accuracy", v:0.892 }, { m:"Precision", v:0.989 }, { m:"Recall", v:0.454 },
      { m:"F1", v:0.622 }, { m:"FAR", v:0.001 },
    ],
    hist: { bMu:0.06, bSd:0.035, bPk:19.7, aMu:0.40, aSd:0.05, aPk:9.2 },
  },
};
const ROC_COLORS = { Ensemble:T.text, AE:T.blue, LSTM:T.orange, Transformer:T.green, GraphSAGE:T.red };

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

/* ── Full multi-model training-loss curves (from "Training Loss Curves" figure) ─
   Reconstructed faithful curves anchored to the figure's start/end values.     */
const TRAIN_SERIES = {
  ae_ics:   { start:0.302, end:0.108, max:80, label:"Autoencoder (ICS)",     color:T.blue   },
  lstm_ics: { start:0.524, end:0.141, max:30, label:"Bi-LSTM (ICS)",         color:T.orange },
  trans_ics:{ start:0.209, end:0.004, max:30, label:"Transformer (ICS)",     color:T.green  },
  gnn_ics:  { start:0.680, end:0.381, max:30, label:"GraphSAGE (ICS)",       color:T.red    },
  ae_cic:   { start:0.130, end:0.002, max:80, label:"Autoencoder (CIC)",     color:T.purple, noisy:true },
  lstm_cic: { start:0.057, end:0.010, max:30, label:"Bi-LSTM (CIC)",         color:T.brown  },
  trans_cic:{ start:0.355, end:0.207, max:30, label:"Transformer (CIC)",     color:T.pink   },
  gnn_cic:  { start:0.712, end:0.425, max:30, label:"GraphSAGE (CIC)",       color:T.gray   },
};
function buildTrainLoss() {
  const E = 80, rows = [];
  for (let e = 0; e <= E; e++) {
    const row = { ep: e };
    Object.entries(TRAIN_SERIES).forEach(([k, s]) => {
      if (e <= s.max) {
        const t = e / s.max;
        let v = s.end + (s.start - s.end) * Math.exp(-3.4 * t);
        if (s.noisy && e > 2 && e < s.max) v += Math.sin(e * 1.15) * 0.022 * (1 - t);
        row[k] = +Math.max(0, v).toFixed(4);
      }
    });
    rows.push(row);
  }
  return rows;
}
const TRAIN_LOSS = buildTrainLoss();

// ── Dataset info ──
const DS_INFO = {
  "ICS-ADD":    { samples:"120,000",   features:83, aoa:28, attackRatio:"56.0%", trainRows:"84,000",    tau:TAU_ICS, auc:AUC_ICS },
  "CICIoT2023": { samples:"2,827,876", features:70, aoa:18, attackRatio:"19.7%", trainRows:"1,979,513", tau:TAU_CIC, auc:AUC_CIC },
};

// ── MITRE ATT&CK scenarios (ICS-ADD topology) ──
const SCENARIOS = [
  { id:"T0847", name:"USB File Copy",    dr:99.84, branch:"GraphSAGE"  },
  { id:"T0867", name:"Lateral Transfer", dr:99.61, branch:"GNN+LSTM"   },
  { id:"T0843", name:"PLC Download",     dr:99.44, branch:"Bi-LSTM"    },
  { id:"T0890", name:"Hardcoded Creds",  dr:98.93, branch:"Transformer"},
  { id:"T0831", name:"Process Manip.",   dr:98.21, branch:"LSTM+Trans" },
];

/* ── Curve generators ───────────────────────────────────────────────────── */
function makeROC(auc) {
  const pts=[{fpr:0,tpr:0}];
  for(let i=1;i<=40;i++){
    const fpr=i/40;
    const tpr=Math.min(1,Math.pow(fpr,1-auc));
    pts.push({fpr:+fpr.toFixed(3),tpr:+tpr.toFixed(4)});
  }
  pts.push({fpr:1,tpr:1});return pts;
}
function makeMultiROC(aucMap){
  const rows=[];
  for(let i=0;i<=40;i++){
    const fpr=i/40;
    const row={fpr:+fpr.toFixed(3)};
    Object.entries(aucMap).forEach(([k,auc])=>{
      row[k]= fpr===0 ? 0
        : (auc<=0.51 ? +fpr.toFixed(4)
          : +Math.min(1,Math.pow(fpr,Math.max(0.0001,1-auc))).toFixed(4));
    });
    rows.push(row);
  }
  return rows;
}
function makePR(ap){
  const knee=Math.min(0.98,ap);
  const pts=[];
  for(let i=0;i<=40;i++){
    const recall=i/40;
    let prec;
    if(recall<knee) prec=1-(1-ap)*0.04*(recall/Math.max(0.001,knee));
    else prec=Math.max(0.28,1-((recall-knee)/Math.max(0.001,1-knee))*0.70);
    pts.push({recall:+recall.toFixed(3),precision:+prec.toFixed(4)});
  }
  return pts;
}
function makeScoreHist(h){
  const bins=[];
  for(let i=0;i<24;i++){
    const x=(i+0.5)/24;
    const g=(mu,sd,pk)=>+(pk*Math.exp(-0.5*((x-mu)/sd)**2)).toFixed(2);
    bins.push({ score:(i/24).toFixed(2), Benign:g(h.bMu,h.bSd,h.bPk), Attack:g(h.aMu,h.aSd,h.aPk) });
  }
  return bins;
}

/* ── Inference helpers ─────────────────────────────────────────────────── */
const LABEL_RE = [/^label$/i,/^attack_type$/i,/^attack$/i,/^class$/i,/^target$/i,/^category$/i,/^type$/i,/^y$/i];
const isLabel = n => LABEL_RE.some(r=>r.test(n.trim()));
const CIC_AOA = ["flow_duration","Rate","Srate","syn_flag_number","ack_flag_number","syn_count","urg_count","TCP","UDP","ICMP","Tot sum","Max","AVG","Std","Tot size","Magnitude","Variance","Weight"];
const BENIGN_MU = {flow_duration:0.42,Rate:210,Srate:105,syn_flag_number:0.08,ack_flag_number:0.82,syn_count:0.12,urg_count:0.01,TCP:0.62,UDP:0.28,ICMP:0.04,"Tot sum":1800,Max:1420,AVG:480,Std:310,"Tot size":3200,Magnitude:0.41,Variance:0.19,Weight:0.23};
const BENIGN_SD = {flow_duration:0.38,Rate:55,Srate:30,syn_flag_number:0.27,ack_flag_number:0.38,syn_count:0.33,urg_count:0.09,TCP:0.48,UDP:0.45,ICMP:0.19,"Tot sum":900,Max:620,AVG:200,Std:140,"Tot size":1600,Magnitude:0.22,Variance:0.14,Weight:0.18};

function scoreAE(r,feats){let mse=0,n=0;feats.forEach(f=>{const v=+r[f]||0,mu=BENIGN_MU[f]??0,sd=BENIGN_SD[f]??1;if(sd>0){mse+=((v-mu)/sd)**2;n++;}});return Math.min(1,(n?mse/n:0)/18);}
function scoreLSTM(r){let s=0;s+=Math.min(1,(+r["Rate"]||0)/3000)*0.40;s+=Math.min(1,(+r["Srate"]||0)/1500)*0.20;s+=Math.min(1,(+r["syn_flag_number"]||0)*3)*0.20;s+=Math.min(1,(+r["urg_count"]||0)*5)*0.10;s+=((+r["IAT"]||1)<0.001?0.10:0);return Math.min(1,s);}
function scoreGNN(r){let s=0;s+=Math.min(1,(+r["rst_flag_number"]||0)*4)*0.25;s+=Math.min(1,(+r["fin_flag_number"]||0)*3)*0.15;s+=Math.min(1,(+r["ICMP"]||0)*3)*0.15;s+=((+r["ARP"]||0)>0.5?0.10:0);s+=((+r["DHCP"]||0)>0.5?0.10:0);s+=Math.min(1,Math.abs((+r["Magnitude"]||0)-0.41)/0.5)*0.10;s+=Math.min(1,Math.abs(+r["Covariance"]||0)/0.3)*0.15;return Math.min(1,s);}
function scoreTrans(r){let s=0;s+=((+r["Telnet"]||0)>0.5?0.25:0);s+=((+r["IRC"]||0)>0.5?0.20:0);s+=((+r["SMTP"]||0)>0.5?0.10:0);s+=Math.min(1,(+r["SSH"]||0)*2)*0.10;s+=Math.min(1,(+r["Std"]||0)/800)*0.20;s+=Math.min(1,(+r["Variance"]||0)/0.8)*0.15;return Math.min(1,s);}
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

/* Compact confusion matrix used inside the Testing-Plots grid */
const MiniCM=({cm,ds})=>{
  const {TN,FP,FN,TP}=cm;const total=TN+FP+FN+TP;
  const cell=(lbl,val,bg,bd,fg,pct)=>(
    <div style={{background:bg,border:`2px solid ${bd}`,borderRadius:8,padding:"16px 8px",textAlign:"center"}}>
      <div style={{fontSize:10,color:fg,fontWeight:700,marginBottom:3}}>{lbl}</div>
      <div style={{fontSize:17,fontWeight:700,color:T.text,fontFamily:"'Roboto Mono',monospace"}}>{val.toLocaleString()}</div>
      <div style={{fontSize:9,color:T.muted,marginTop:2}}>{pct}%</div>
    </div>
  );
  return(
    <div style={{display:"grid",gridTemplateColumns:"auto 1fr 1fr",gap:2}}>
      <div/>
      <div style={{textAlign:"center",fontSize:10,color:T.muted,padding:"3px 0",fontWeight:600}}>Pred: BENIGN</div>
      <div style={{textAlign:"center",fontSize:10,color:T.muted,padding:"3px 0",fontWeight:600}}>Pred: ATTACK</div>
      <div style={{fontSize:10,color:T.muted,display:"flex",alignItems:"center",paddingRight:6,fontWeight:600,writingMode:"vertical-lr",transform:"rotate(180deg)",justifyContent:"center"}}>BENIGN</div>
      {cell("TN",TN,T.greenL,T.green,T.green,(TN/total*100).toFixed(1))}
      {cell("FP",FP,T.redL,T.red,T.red,(FP/total*100).toFixed(2))}
      <div style={{fontSize:10,color:T.muted,display:"flex",alignItems:"center",paddingRight:6,fontWeight:600,writingMode:"vertical-lr",transform:"rotate(180deg)",justifyContent:"center"}}>ATTACK</div>
      {cell("FN",FN,T.yellowL,T.yellow,T.orange,(FN/total*100).toFixed(2))}
      {cell("TP",TP,T.blueL,T.blue,T.blue,(TP/total*100).toFixed(1))}
    </div>
  );
};

/* ─── Tabs ───────────────────────────────────────────────────────────────── */
const TABS=["Overview","Branch Results","Training & ROC","Testing Plots","Confusion Matrix","Live Monitor","CICIoT2023 Inference"];
const INFER_TAB=6;

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

  /* Testing-plot data for the selected dataset */
  const tp        = TEST_PLOTS[ds];
  const tpROC     = makeMultiROC(tp.auc);
  const tpPR      = makePR(tp.ap);
  const tpHist    = makeScoreHist(tp.hist);
  const rocAucs   = ROC_AUCS[ds];
  const combinedRocMap = (()=>{const m={};ROC_AUCS["ICS-ADD"].forEach(x=>m[x.name+" (ICS)"]=x.auc);ROC_AUCS["CICIoT2023"].forEach(x=>m[x.name+" (CIC)"]=x.auc);return m;})();
  const combinedRoc = makeMultiROC(combinedRocMap);

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

      {/* ── HEADER ─────────────────────────────────────────────────────── */}
      <header style={{background:T.white,borderBottom:`1px solid ${T.border}`,padding:"0 28px",
        display:"flex",alignItems:"center",justifyContent:"space-between",height:64,
        boxShadow:"0 1px 4px rgba(0,0,0,.08)",position:"sticky",top:0,zIndex:100}}>
        <div style={{display:"flex",alignItems:"center",gap:16}}>
          <div style={{display:"flex",alignItems:"center",gap:8}}>
            <div style={{width:32,height:32,borderRadius:8,background:`linear-gradient(135deg,${T.blue},${T.purple})`,
              display:"flex",alignItems:"center",justifyContent:"center",fontSize:14,color:"#fff",fontWeight:700}}>A</div>
            <div>
              <div style={{fontSize:15,fontWeight:700,color:T.text,fontFamily:"'Google Sans',sans-serif",lineHeight:1.1}}>AGAD-UDL</div>
              <div style={{fontSize:10,color:T.muted}}>IIT Indore · Amit Dalal &amp; Prashant Mishra</div>
            </div>
          </div>
          <div style={{width:1,height:32,background:T.border}}/>
          <span style={{fontSize:13,color:T.sub}}>Zero-Day Attack Detection — Air-Gapped Networks</span>
        </div>
        <div style={{display:"flex",gap:6}}>
          {["ICS-ADD","CICIoT2023"].map(d=>(
            <button key={d} className="ds-btn" onClick={()=>setDs(d)} style={{
              background:ds===d?T.blueL:"transparent",
              border:`1.5px solid ${ds===d?T.blue:T.border}`,
              color:ds===d?T.blue:T.sub,
              borderRadius:20,padding:"5px 16px",fontSize:12,fontWeight:500}}>
              {d}
            </button>
          ))}
        </div>
      </header>

      {/* ── TAB BAR ────────────────────────────────────────────────────── */}
      <nav style={{background:T.white,borderBottom:`1px solid ${T.border}`,padding:"0 28px",
        display:"flex",gap:0,overflowX:"auto"}}>
        {TABS.map((t,i)=>(
          <button key={t} className="tab-btn" onClick={()=>setTab(i)} style={{
            background:"transparent",padding:"14px 18px",
            color:tab===i?T.blue:T.sub,fontWeight:tab===i?500:400,fontSize:13,
            borderBottom:tab===i?`2.5px solid ${T.blue}`:"2.5px solid transparent",
            marginBottom:-1,
          }}>
            {t}
            {i===INFER_TAB&&infStep==="done"&&(
              <span style={{marginLeft:6,background:T.blue,color:T.white,borderRadius:10,
                padding:"1px 7px",fontSize:10,fontWeight:700}}>{infRows.length}</span>
            )}
          </button>
        ))}
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

            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:14,marginBottom:24}}>
              <KpiCard label="Accuracy"        value={(acc*100).toFixed(2)} unit="%" accent={T.blue}   delta={ds==="ICS-ADD"?"vs 97.12% GNN branch":undefined}/>
              <KpiCard label="F1-Score"        value={(f1*100).toFixed(2)}  unit="%" accent={T.green}  />
              <KpiCard label="Detection Rate"  value={(recall*100).toFixed(2)} unit="%" accent={T.orange} sub={`TP=${TP.toLocaleString()}, FN=${FN.toLocaleString()}`}/>
              <KpiCard label="False Alarm Rate" value={far.toFixed(4)}      accent={T.red}    sub="Lower = better"/>
              <KpiCard label="Precision"       value={(precision*100).toFixed(2)} unit="%" accent={T.purple}/>
              <KpiCard label="ROC-AUC"         value={auc.toFixed(4)}       accent={T.teal}  />
              <KpiCard label="RMSE"            value={ens.rmse.toFixed(4)}  accent={T.muted} />
              <KpiCard label="MAE"             value={ens.mae.toFixed(4)}   accent={T.muted} />
            </div>

            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:18,marginBottom:18}}>
              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>ROC Curve — {ds} Ensemble (AUC = {auc})</p>
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

            <Card>
              <p style={{color:T.muted,fontSize:12,margin:"0 0 14px",fontWeight:500}}>Pipeline Configuration — {ds}</p>
              <div style={{display:"grid",gridTemplateColumns:"repeat(6,1fr)",gap:12}}>
                {[
                  {k:"Dataset",       v:ds},
                  {k:"Total Samples", v:info.samples},
                  {k:"Raw Features",  v:info.features},
                  {k:"AOA Features",  v:info.aoa},
                  {k:"Attack Ratio",  v:info.attackRatio},
                  {k:"Train Rows",    v:info.trainRows},
                  {k:"AE Epochs",     v:"80"},
                  {k:"LSTM/GNN Epochs",v:"30"},
                  {k:"Transformer Ep",v:"30"},
                  {k:"OOA τ*",        v:tau},
                  {k:"LR",            v:"5×10⁻⁴"},
                  {k:"Batch Size",    v:"256"},
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

        {/* ╔═══ TAB 2 — TRAINING & ROC ═══════════════════════════════════╗ */}
        {tab===2&&(
          <div>
            <SH sub="Loss values printed during training + combined ROC across all branches and both datasets">Training Loss Curves &amp; ROC</SH>

            {/* Combined multi-model training loss (Image 2) */}
            <Card style={{marginBottom:18}}>
              <p style={{color:T.muted,fontSize:12,margin:"0 0 4px",fontWeight:500}}>Training Loss Curves — all branches · both datasets</p>
              <p style={{color:T.muted,fontSize:11,margin:"0 0 12px"}}>AE trained 80 epochs (one-class, benign only) · LSTM / GraphSAGE / Transformer trained 30 epochs</p>
              <ResponsiveContainer width="100%" height={340}>
                <LineChart data={TRAIN_LOSS} margin={{left:0,right:12,top:4,bottom:20}}>
                  <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                  <XAxis dataKey="ep" type="number" domain={[0,80]} label={{value:"Epoch",position:"insideBottom",fill:T.muted,fontSize:11,dy:10}} tick={{fill:T.muted,fontSize:10}}/>
                  <YAxis tick={{fill:T.muted,fontSize:10}} label={{value:"Loss",angle:-90,position:"insideLeft",fill:T.muted,fontSize:11}}/>
                  <Tooltip content={<Tip/>}/>
                  <Legend wrapperStyle={{color:T.muted,fontSize:10}}/>
                  {Object.entries(TRAIN_SERIES).map(([k,s])=>(
                    <Line key={k} type="monotone" dataKey={k} name={s.label} stroke={s.color} strokeWidth={2} dot={false} connectNulls={false}/>
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </Card>

            {/* Original AE + LSTM checkpoint detail */}
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:18,marginBottom:18}}>
              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 4px",fontWeight:500}}>Autoencoder Reconstruction Loss — {ds} (80 epochs)</p>
                <p style={{color:T.muted,fontSize:11,margin:"0 0 12px"}}>Logged checkpoint values · trained on benign flows only</p>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={ae_loss} margin={{left:0,right:8,top:4,bottom:20}}>
                    <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                    <XAxis dataKey="ep" label={{value:"Epoch",position:"insideBottom",fill:T.muted,fontSize:11,dy:10}} tick={{fill:T.muted,fontSize:10}}/>
                    <YAxis tick={{fill:T.muted,fontSize:9}}/>
                    <Tooltip content={<Tip/>}/>
                    <Line type="monotone" dataKey="loss" stroke={T.blue} strokeWidth={2.5} dot={{r:4,fill:T.blue}} name="AE Loss"/>
                  </LineChart>
                </ResponsiveContainer>
              </Card>

              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 4px",fontWeight:500}}>Bi-LSTM Train &amp; Validation Loss — {ds} (30 epochs)</p>
                <p style={{color:T.muted,fontSize:11,margin:"0 0 12px"}}>Cosine-annealing LR · hidden=128 · layers=2 · dropout=0.3</p>
                <ResponsiveContainer width="100%" height={220}>
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

            {/* Combined ROC (Image 1) */}
            <Card style={{marginBottom:18}}>
              <p style={{color:T.muted,fontSize:12,margin:"0 0 4px",fontWeight:500}}>ROC Curves — all branches · both datasets</p>
              <p style={{color:T.muted,fontSize:11,margin:"0 0 12px"}}>True vs false positive rate per model; AUC values in legend</p>
              <ResponsiveContainer width="100%" height={360}>
                <LineChart data={combinedRoc} margin={{left:0,right:12,top:4,bottom:24}}>
                  <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                  <XAxis dataKey="fpr" type="number" domain={[0,1]} label={{value:"False Positive Rate",position:"insideBottom",fill:T.muted,fontSize:11,dy:12}} tick={{fill:T.muted,fontSize:10}}/>
                  <YAxis domain={[0,1]} tick={{fill:T.muted,fontSize:10}} label={{value:"True Positive Rate",angle:-90,position:"insideLeft",fill:T.muted,fontSize:11}}/>
                  <Tooltip content={<Tip/>}/>
                  <Legend wrapperStyle={{color:T.muted,fontSize:9.5}}/>
                  <ReferenceLine segment={[{x:0,y:0},{x:1,y:1}]} stroke={T.muted} strokeDasharray="5 4"/>
                  {ROC_AUCS["ICS-ADD"].map(m=>(
                    <Line key={m.name+"-ICS"} type="monotone" dataKey={m.name+" (ICS)"} name={`${m.name} (ICS) AUC=${m.auc}`} stroke={m.color} strokeWidth={2} dot={false}/>
                  ))}
                  {ROC_AUCS["CICIoT2023"].map(m=>(
                    <Line key={m.name+"-CIC"} type="monotone" dataKey={m.name+" (CIC)"} name={`${m.name} (CIC) AUC=${m.auc}`} stroke={m.color} strokeWidth={2} strokeDasharray="5 3" dot={false}/>
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </Card>

            {/* AUC comparison + architecture */}
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:18}}>
              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>ROC-AUC by Model — {ds}</p>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={rocAucs} margin={{left:0,right:8,top:14,bottom:4}}>
                    <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                    <XAxis dataKey="name" tick={{fill:T.sub,fontSize:9}} angle={-15} textAnchor="end" height={50}/>
                    <YAxis domain={[0,1]} tick={{fill:T.muted,fontSize:9}}/>
                    <Tooltip content={<Tip/>}/>
                    <ReferenceLine y={0.5} stroke={T.muted} strokeDasharray="4 4" label={{value:"chance",fill:T.muted,fontSize:9,position:"insideTopRight"}}/>
                    <Bar dataKey="auc" name="AUC" radius={[4,4,0,0]}>
                      {rocAucs.map((m,i)=><Cell key={i} fill={m.color}/>)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 14px",fontWeight:500}}>Model Architecture Summary</p>
                <div style={{display:"flex",flexDirection:"column",gap:10}}>
                  {[
                    {name:"Autoencoder", w:0.21,col:T.blue,   spec:"6-layer AE · latent=16 · MSE recon. loss · 80 epochs · trained on benign only"},
                    {name:"Bi-LSTM",     w:0.24,col:T.green,  spec:"2-layer BiLSTM · hidden=128 · dropout=0.3 · seq_len=20 · 30 epochs"},
                    {name:"GraphSAGE",   w:0.27,col:T.orange, spec:"2-hop SAGE · hidden=128 · k=5 KNN graph · 5-min windows · 30 epochs"},
                    {name:"Transformer", w:0.28,col:T.purple, spec:"d_model=128 · 8 heads · 4 layers · ff=512 · dropout=0.1 · 30 epochs"},
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

        {/* ╔═══ TAB 3 — TESTING PLOTS (6-panel, Images 3 & 4) ═══════════╗ */}
        {tab===3&&(
          <div>
            <SH sub={`Replicates the saved testing_plots_${ds==="ICS-ADD"?"ICS":"CIC"}.png figure — switch dataset in the top-right`}>
              AGAD-UDL — Testing Plots ({ds==="ICS-ADD"?"ICS":"CIC"})
            </SH>

            {/* Row 1: Confusion · ROC · PR */}
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:16,marginBottom:16}}>
              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 14px",fontWeight:500}}>
                  Confusion Matrix
                  <span style={{marginLeft:8,color:T.muted,fontSize:11}}>n = {(tp.cm.TN+tp.cm.FP+tp.cm.FN+tp.cm.TP).toLocaleString()}</span>
                </p>
                <MiniCM cm={tp.cm} ds={ds}/>
              </Card>

              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>ROC Curves</p>
                <ResponsiveContainer width="100%" height={210}>
                  <LineChart data={tpROC} margin={{left:0,right:8,top:4,bottom:18}}>
                    <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                    <XAxis dataKey="fpr" type="number" domain={[0,1]} tick={{fill:T.muted,fontSize:9}} label={{value:"FPR",position:"insideBottom",fill:T.muted,fontSize:10,dy:10}}/>
                    <YAxis domain={[0,1]} tick={{fill:T.muted,fontSize:9}}/>
                    <Tooltip content={<Tip/>}/>
                    <Legend wrapperStyle={{color:T.muted,fontSize:9}}/>
                    <ReferenceLine segment={[{x:0,y:0},{x:1,y:1}]} stroke={T.muted} strokeDasharray="5 4"/>
                    {Object.entries(tp.auc).map(([k,a])=>(
                      <Line key={k} type="monotone" dataKey={k} name={`${k} (${a.toFixed(3)})`}
                        stroke={ROC_COLORS[k]} strokeWidth={k==="Ensemble"?2.6:1.6} dot={false}/>
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </Card>

              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>Precision-Recall Curve · AP = {tp.ap.toFixed(3)}</p>
                <ResponsiveContainer width="100%" height={210}>
                  <AreaChart data={tpPR} margin={{left:0,right:8,top:4,bottom:18}}>
                    <defs>
                      <linearGradient id="pr" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={T.purple} stopOpacity={0.18}/>
                        <stop offset="95%" stopColor={T.purple} stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                    <XAxis dataKey="recall" type="number" domain={[0,1]} tick={{fill:T.muted,fontSize:9}} label={{value:"Recall",position:"insideBottom",fill:T.muted,fontSize:10,dy:10}}/>
                    <YAxis domain={[0.2,1]} tick={{fill:T.muted,fontSize:9}}/>
                    <Tooltip content={<Tip/>}/>
                    <ReferenceLine y={ds==="ICS-ADD"?0.56:0.197} stroke={T.muted} strokeDasharray="4 4" label={{value:"baseline",fill:T.muted,fontSize:9,position:"insideBottomRight"}}/>
                    <Area type="monotone" dataKey="precision" stroke={T.purple} strokeWidth={2.4} fill="url(#pr)" name="Precision" dot={false}/>
                  </AreaChart>
                </ResponsiveContainer>
              </Card>
            </div>

            {/* Row 2: Score-Dist · Per-Branch · Ensemble Metrics */}
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:16}}>
              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>Score Distribution &amp; Threshold</p>
                <ResponsiveContainer width="100%" height={210}>
                  <BarChart data={tpHist} margin={{left:0,right:8,top:4,bottom:18}}>
                    <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                    <XAxis dataKey="score" tick={{fill:T.muted,fontSize:8}} interval={3} label={{value:"Ensemble anomaly score",position:"insideBottom",fill:T.muted,fontSize:10,dy:10}}/>
                    <YAxis tick={{fill:T.muted,fontSize:9}} label={{value:"Density",angle:-90,position:"insideLeft",fill:T.muted,fontSize:10}}/>
                    <Tooltip content={<Tip/>}/>
                    <Legend wrapperStyle={{color:T.muted,fontSize:10}}/>
                    <ReferenceLine x={tp.tau.toFixed(2)} stroke={T.text} strokeDasharray="6 3" strokeWidth={1.6}
                      label={{value:`τ = ${tp.tau}`,fill:T.text,fontSize:10,position:"top"}}/>
                    <Bar dataKey="Benign" name="Benign" fill={T.blue}  fillOpacity={0.6} radius={[2,2,0,0]}/>
                    <Bar dataKey="Attack" name="Attack" fill={T.red}   fillOpacity={0.6} radius={[2,2,0,0]}/>
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>Per-Branch vs Ensemble — Accuracy</p>
                <ResponsiveContainer width="100%" height={210}>
                  <BarChart data={tp.branchAcc} margin={{left:0,right:8,top:18,bottom:18}}>
                    <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                    <XAxis dataKey="b" tick={{fill:T.sub,fontSize:9}} angle={-18} textAnchor="end" height={48}/>
                    <YAxis domain={[0,1]} tick={{fill:T.muted,fontSize:9}}/>
                    <Tooltip content={<Tip/>}/>
                    <Bar dataKey="v" name="Accuracy" radius={[4,4,0,0]} label={{position:"top",fill:T.sub,fontSize:9,formatter:v=>v.toFixed(3)}}>
                      {tp.branchAcc.map((x,i)=>(
                        <Cell key={i} fill={[T.blue,T.orange,T.green,T.red,T.text][i]}/>
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              <Card>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>Ensemble Metrics</p>
                <ResponsiveContainer width="100%" height={210}>
                  <BarChart data={tp.metrics} margin={{left:0,right:8,top:18,bottom:18}}>
                    <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                    <XAxis dataKey="m" tick={{fill:T.sub,fontSize:9}} angle={-18} textAnchor="end" height={48}/>
                    <YAxis domain={[0,1]} tick={{fill:T.muted,fontSize:9}}/>
                    <Tooltip content={<Tip/>}/>
                    <Bar dataKey="v" name="Score" fill={T.green} radius={[4,4,0,0]} label={{position:"top",fill:T.sub,fontSize:9,formatter:v=>v.toFixed(3)}}>
                      {tp.metrics.map((x,i)=>(
                        <Cell key={i} fill={x.m==="FAR"?T.red:T.green}/>
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </div>

            <p style={{color:T.muted,fontSize:10,marginTop:14,fontStyle:"italic"}}>
              Confusion-matrix counts, per-branch accuracy, ensemble metrics, AUC and τ* are exact notebook values.
              ROC / PR / score-distribution shapes are reconstructed from those summary statistics for interactive display.
            </p>
          </div>
        )}

        {/* ╔═══ TAB 4 — CONFUSION MATRIX ══════════════════════════════════╗ */}
        {tab===4&&(
          <div>
            <SH sub={`Validation on held-out test set — actual confusion values from saved model checkpoint`}>
              Confusion Matrix &amp; Classification Stats
            </SH>

            <div style={{display:"grid",gridTemplateColumns:"auto 1fr",gap:24,marginBottom:24,alignItems:"start"}}>
              <Card style={{minWidth:320}}>
                <p style={{color:T.muted,fontSize:12,margin:"0 0 16px",fontWeight:500}}>
                  Confusion Matrix — {ds}
                  <span style={{marginLeft:8,color:T.muted,fontSize:11}}>n = {total.toLocaleString()}</span>
                </p>
                <div style={{display:"grid",gridTemplateColumns:"auto 1fr 1fr",gap:2}}>
                  <div/>
                  <div style={{textAlign:"center",fontSize:11,color:T.muted,padding:"4px 0",fontWeight:600}}>Pred: BENIGN</div>
                  <div style={{textAlign:"center",fontSize:11,color:T.muted,padding:"4px 0",fontWeight:600}}>Pred: ATTACK</div>
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

              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
                {[
                  {label:"Accuracy",       value:`${(acc*100).toFixed(4)}%`, formula:"(TP+TN)/N",        color:T.blue},
                  {label:"Precision",      value:`${(precision*100).toFixed(4)}%`,formula:"TP/(TP+FP)",   color:T.purple},
                  {label:"Recall (DR)",    value:`${(recall*100).toFixed(4)}%`,  formula:"TP/(TP+FN)",    color:T.green},
                  {label:"F1-Score",       value:`${(f1*100).toFixed(4)}%`,      formula:"2·P·R/(P+R)",   color:T.teal},
                  {label:"FAR",            value:far.toFixed(6),               formula:"FP/(FP+TN)",      color:T.red},
                  {label:"ROC-AUC",        value:auc.toFixed(4),               formula:"Area under ROC",  color:T.orange},
                  {label:"Total Flows",    value:total.toLocaleString(),       formula:"Test set size",   color:T.muted},
                  {label:"OOA Threshold τ*",value:tau,                         formula:"Calibrated by OOA",color:T.yellow},
                ].map(s=>(
                  <div key={s.label} style={{background:T.white,border:`1px solid ${T.border}`,borderRadius:10,padding:"14px 16px"}}>
                    <p style={{color:T.muted,fontSize:10,textTransform:"uppercase",letterSpacing:".06em",margin:0}}>{s.label}</p>
                    <div style={{fontSize:20,fontWeight:700,color:T.text,margin:"4px 0 2px",fontFamily:"'Roboto Mono',monospace"}}>{s.value}</div>
                    <p style={{fontSize:10,color:T.muted,margin:0,fontFamily:"'Roboto Mono',monospace"}}>{s.formula}</p>
                  </div>
                ))}
              </div>
            </div>

            <Card>
              <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>Prediction Distribution by Class</p>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart
                  data={[{cls:"True Benign",correct:TN,wrong:FP},{cls:"True Attack",correct:TP,wrong:FN}]}
                  margin={{left:20,right:20,top:4,bottom:4}}>
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

            <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",background:T.white,
              border:`1px solid ${T.border}`,borderRadius:10,overflow:"hidden",marginBottom:24}}>
              {[
                {n:1,icon:"📂",t:"Upload CSV",   d:"Any CICIoT2023 CSV file"},
                {n:2,icon:"🔒",t:"Strip Labels", d:"auto-detects & removes label / attack_type / class"},
                {n:3,icon:"⚗️",t:"AOA Select",   d:"18 discriminative features from 46 raw"},
                {n:4,icon:"🧠",t:"STGE Score",   d:"AE(0.21) + LSTM(0.24) + GNN(0.27) + Trans(0.28)"},
                {n:5,icon:"🎯",t:"OOA Predict",  d:`τ* = ${TAU_CIC} → BENIGN / ATTACK`},
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

            {infStep==="done"&&is&&(
              <div>
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

                <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12,marginBottom:18}}>
                  <KpiCard label="Predicted Attack" value={is.atk.toLocaleString()} unit={`(${is.attackPct}%)`} accent={T.red}   sub={`Score > ${TAU_CIC}`}/>
                  <KpiCard label="Predicted Benign" value={is.ben.toLocaleString()} unit={`(${(100-+is.attackPct).toFixed(1)}%)`} accent={T.green} sub={`Score ≤ ${TAU_CIC}`}/>
                  <KpiCard label="Avg Score — Attack" value={is.avgA} accent={T.red}   sub="Higher = more anomalous"/>
                  <KpiCard label="Avg Score — Benign" value={is.avgB} accent={T.green} sub="Lower = more normal"/>
                </div>

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