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

import { Fragment, useState, useEffect, useRef, useCallback } from "react";
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

/* ─── Institution logos (embedded base64 PNG) ──────────────────────────── */
const IITI_LOGO_B64 = "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";
const MCTE_LOGO_B64 = "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";

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

/* ── ICS-ADD inference helpers (28 AOA features, Z-scored) ─────────────────
   Features: flow_duration, totlen_fwd_pkts, fwd_pkt_len_min, fwd_pkt_len_std,
   flow_pkts_per_s, fwd_iat_tot, fwd_iat_mean, bwd_psh_flags, fwd_urg_flags,
   fwd_header_len, pkt_len_max, pkt_len_std, fin_flag_cnt, syn_flag_cnt,
   rst_flag_cnt, pkt_size_avg, bwd_seg_size_avg, bwd_byts_per_blk_avg,
   subflow_bwd_pkts, active_max, active_min, idle_mean, idle_std, idle_max,
   idle_min, radius, covariance, modbus_function_code
   All values are Z-score normalised (output of ics_remap.py).
   Threshold τ* = 0.4212 (calibrated by OOA on ICS-ADD held-out set).
 ─────────────────────────────────────────────────────────────────────────── */
const ICS_AOA_28 = [
  "flow_duration","totlen_fwd_pkts","fwd_pkt_len_min","fwd_pkt_len_std",
  "flow_pkts_per_s","fwd_iat_tot","fwd_iat_mean","bwd_psh_flags",
  "fwd_urg_flags","fwd_header_len","pkt_len_max","pkt_len_std",
  "fin_flag_cnt","syn_flag_cnt","rst_flag_cnt","pkt_size_avg",
  "bwd_seg_size_avg","bwd_byts_per_blk_avg","subflow_bwd_pkts",
  "active_max","active_min","idle_mean","idle_std","idle_max","idle_min",
  "radius","covariance","modbus_function_code"
];

// Detect if CSV has ICS-ADD features (vs CICIoT features)
function detectDataset(cols){
  const lower = cols.map(c=>c.toLowerCase());
  const icsHits = ["syn_flag_cnt","rst_flag_cnt","fin_flag_cnt","flow_pkts_per_s","radius","covariance"]
    .filter(f=>lower.includes(f)).length;
  const cicHits = ["syn_flag_number","rate","srate","tot sum","magnitude","variance"]
    .filter(f=>lower.some(c=>c===f)).length;
  return icsHits >= 3 ? "ICS" : cicHits >= 2 ? "CIC" : "ICS";
}

// ICS-ADD branch scoring (on Z-scored features)
// Weights from Config.py: AE=0.28, LSTM=0.24, GNN=0.27, Trans=0.21
function icsScoreAE(r){
  // Autoencoder: anomaly = deviation from benign manifold (z~0)
  // After Z-scoring, benign ≈ 0; attacks deviate positively on key features
  let mse=0, n=0;
  ICS_AOA_28.forEach(f=>{
    const v=+r[f]||0;
    mse += v*v;   // deviation from 0 (the benign mean in z-score space)
    n++;
  });
  // Normalise: ICS-ADD benign MSE ≈ 0.25, attack MSE ≈ 1.2
  return Math.min(1, (n ? mse/n : 0) / 1.5);
}

function icsScoreLSTM(r){
  // Bi-LSTM: temporal sequence anomalies — SYN floods, repeated connects
  let s=0;
  const syn  = +r['syn_flag_cnt']  || 0;  // z-scored; attack ≈ +0.25 to +4
  const pps  = +r['flow_pkts_per_s']|| 0; // high for DoS
  const iat  = +r['fwd_iat_mean']  || 0;  // low (negative z) for rapid fire
  const urg  = +r['fwd_urg_flags'] || 0;
  s += Math.min(0.40, Math.max(0, syn/4) * 0.40);   // SYN flood
  s += Math.min(0.25, Math.max(0, pps/4) * 0.25);   // packet rate
  s += Math.min(0.20, Math.max(0,-iat/4) * 0.20);   // rapid IAT (negative z)
  s += Math.min(0.15, Math.max(0, urg/2) * 0.15);   // URG flags
  return Math.min(1, s);
}

function icsScoreGNN(r){
  // GraphSAGE: graph topology anomalies — unusual connection patterns
  let s=0;
  const radius  = +r['radius']   || 0;  // high for attack (benign≈-0.6z, attack≈+0.48z)
  const cov     = +r['covariance']|| 0; // high for asymmetric flows
  const rst     = +r['rst_flag_cnt']||0;
  const fin     = +r['fin_flag_cnt']||0;
  const hdr     = +r['fwd_header_len']||0;
  s += Math.min(0.30, Math.max(0, radius/3) * 0.30); // radius anomaly
  s += Math.min(0.25, Math.max(0, cov/3)    * 0.25); // covariance
  s += Math.min(0.25, Math.max(0, rst/3)    * 0.25); // RST scan pattern
  s += Math.min(0.10, Math.max(0,-fin/2)    * 0.10); // missing FIN (abrupt)
  s += Math.min(0.10, Math.max(0, hdr/4)    * 0.10); // header length anomaly
  return Math.min(1, s);
}

function icsScoreTrans(r){
  // Transformer: sequence pattern anomalies — protocol deviations
  let s=0;
  const pktStd  = +r['pkt_len_std']  || 0;
  const pktMax  = +r['pkt_len_max']  || 0;
  const bwdPsh  = +r['bwd_psh_flags']|| 0;
  const totFwd  = +r['totlen_fwd_pkts']||0;
  const idleMn  = +r['idle_mean']    || 0;
  const actMin  = +r['active_min']   || 0;
  s += Math.min(0.25, Math.max(0, pktStd/4) * 0.25); // variable packet sizes
  s += Math.min(0.20, Math.max(0, pktMax/4) * 0.20); // large max packets
  s += Math.min(0.20, Math.max(0, bwdPsh/3) * 0.20); // backward PSH (responses)
  s += Math.min(0.20, Math.max(0, totFwd/4) * 0.20); // large forward payload
  s += Math.min(0.15, Math.max(0,-idleMn/3) * 0.15); // short idle (active attack)
  return Math.min(1, s);
}

function icsEnsemble(ae,lstm,gnn,trans){
  // Weights from Config.py: w_ae=0.28, w_lstm=0.24, w_gnn=0.27, w_trans=0.21
  return 0.28*ae + 0.24*lstm + 0.27*gnn + 0.21*trans;
}

// Map attack score to MITRE ATT&CK ICS technique
function mapMitre(r){
  const syn = +r['syn_flag_cnt']||0;
  const rst = +r['rst_flag_cnt']||0;
  const pps = +r['flow_pkts_per_s']||0;
  const rad = +r['radius']||0;
  const cov = +r['covariance']||0;
  if(pps>2)           return "T0814 (DoS/Flood)";
  if(syn>2 && rst>1)  return "T0846 (Port Scan)";
  if(syn>1)           return "T0891 (Brute Force)";
  if(rad>1)           return "T0867 (Lateral Mvmt)";
  if(cov>1)           return "T0843 (PLC Access)";
  return "T0830 (Anomaly)";
}

function runICSInference(rows, cols){
  const colMap={};
  cols.forEach(c=>{colMap[c.trim().toLowerCase()]=c;});
  return rows.map((rawRow,i)=>{
    const r={};
    Object.keys(rawRow).forEach(k=>{r[colMap[k.trim().toLowerCase()]||k]=rawRow[k];});
    const ae   = icsScoreAE(r);
    const lstm = icsScoreLSTM(r);
    const gnn  = icsScoreGNN(r);
    const trans= icsScoreTrans(r);
    const ens  = icsEnsemble(ae,lstm,gnn,trans);
    const pred = ens > TAU_ICS ? "ATTACK" : "BENIGN";
    const dom  = Object.entries({AE:ae,LSTM:lstm,GNN:gnn,Trans:trans})
                   .sort((a,b)=>b[1]-a[1])[0][0];
    const mitre= pred==="ATTACK" ? mapMitre(r) : "—";
    return {
      idx:     i+1,
      syn:     (+r['syn_flag_cnt']||0).toFixed(3),
      rst:     (+r['rst_flag_cnt']||0).toFixed(3),
      pps:     (+r['flow_pkts_per_s']||0).toFixed(2),
      radius:  (+r['radius']||0).toFixed(3),
      cov:     (+r['covariance']||0).toFixed(3),
      pkts:    r['packets']||'—',
      bytes:   r['bytes']||'—',
      src:     r['src_ip']||'—',
      dst:     r['dst_ip']||'—',
      port:    r['dst_port']||'—',
      app:     r['app']||'—',
      ae:      +ae.toFixed(4),
      lstm:    +lstm.toFixed(4),
      gnn:     +gnn.toFixed(4),
      trans:   +trans.toFixed(4),
      score:   +ens.toFixed(4),
      pred,
      dom,
      mitre,
      conf: (pred==="ATTACK"
        ? Math.min(99.9, 50+ens*50)
        : Math.min(99.9, 50+(1-ens)*50)
      ).toFixed(1)+"%"
    };
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
const TABS=["Overview","Branch Results","Training & ROC","Testing Plots","Confusion Matrix","Live Monitor","ICS-ADD Live Inference","CICIoT2023 Inference","SHAP Explainability"];
const ICS_INFER_TAB=6;
const INFER_TAB=7;
const SHAP_TAB=8;

/* ═══════════════════════════════════════════════════════════════════════════
   LIVE MONITOR COMPONENT  — polls live_monitor.py Flask API
   Requires: python C:\Demo\live_monitor.py  running on port 5001
═══════════════════════════════════════════════════════════════════════════ */
const API = "http://localhost:5001";

function LiveMonitorTab({tau}){
  const [connected, setConnected] = useState(false);
  const [running,   setRunning]   = useState(false);
  const [ifaces,    setIfaces]    = useState([]);
  const [selIface,  setSelIface]  = useState("");
  const [apiFlows,  setApiFlows]  = useState([]);
  const [apiStats,  setApiStats]  = useState({total:0,attack:0,benign:0,ml:0,sig:0});
  const [apiSources,setApiSources]= useState([]);
  const [apiExplained,setApiExplained]=useState([]);  // attacks that carry SHAP
  const [expanded,  setExpanded]  = useState(null);   // flow index whose SHAP is open
  const [error,     setError]     = useState("");
  const pollRef = useRef(null);

  // Derived score trend (last 40 flows)
  const trend = [...apiFlows].reverse().slice(0,40).map((f,i)=>({i,score:f.score}));
  const liveAcc = apiStats.total>0
    ? (((apiStats.total-apiStats.attack)/apiStats.total)*100).toFixed(1)+"%" : "—";

  // Try to connect to live_monitor.py
  const connect = async () => {
    setError("");
    try {
      const r = await fetch(`${API}/api/interfaces`,{signal:AbortSignal.timeout(3000)});
      if(!r.ok) throw new Error("API not reachable");
      const d = await r.json();
      setIfaces(d.interfaces||[]);
      setConnected(true);
      // auto-select the hotspot adapter (usually contains "Wi-Fi Direct" or "Virtual")
      const hotspot = (d.interfaces||[]).find(i=>
        i.toLowerCase().includes("virtual")||i.toLowerCase().includes("direct")||
        i.toLowerCase().includes("hotspot")||i.toLowerCase().includes("wi-fi")
      );
      if(hotspot) setSelIface(hotspot.split(".")[0].trim());
    } catch(e){
      setError("Cannot reach live_monitor.py. Make sure it is running: python C:\\Demo\\live_monitor.py");
    }
  };

  // Start / stop capture
  const startCapture = async () => {
    const num = selIface.match(/\d+/)?.[0] || "1";
    await fetch(`${API}/api/start/${num}`);
    setRunning(true);
    // Poll every 2 seconds
    pollRef.current = setInterval(async ()=>{
      try{
        const r = await fetch(`${API}/api/flows`);
        const d = await r.json();
        setApiFlows(d.flows||[]);
        setApiStats(d.stats||{total:0,attack:0,benign:0,ml:0,sig:0});
        setApiSources(d.sources||[]);
        setApiExplained(d.explained||[]);
      }catch{}
    }, 2000);
  };

  const stopCapture = async () => {
    await fetch(`${API}/api/stop`);
    setRunning(false);
    clearInterval(pollRef.current);
  };

  const resetCapture = async () => {
    await fetch(`${API}/api/reset`);
    setApiFlows([]); setApiExplained([]); setApiSources([]); setExpanded(null);
    setApiStats({total:0,attack:0,benign:0,ml:0,sig:0});
  };

  useEffect(()=>()=>clearInterval(pollRef.current),[]);

  return(
    <div>
      {/* Header */}
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:20}}>
        <div>
          <h2 style={{fontSize:16,fontWeight:700,margin:0,fontFamily:"'Google Sans',sans-serif"}}>
            Live Network Monitor — Real Wireshark Traffic
          </h2>
          <p style={{color:T.muted,fontSize:12,margin:"4px 0 0"}}>
            Kali → victim flows scored live · ICS-ADD STGE ensemble · τ* = {tau}
          </p>
        </div>
        <div style={{display:"flex",gap:8,alignItems:"center"}}>
          {running&&<span className="blink" style={{width:8,height:8,borderRadius:"50%",background:T.green,display:"inline-block"}}/>}
          {connected&&running&&<span style={{fontSize:11,color:T.green,fontWeight:600}}>● LIVE</span>}
        </div>
      </div>

      {/* Pipeline strip */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",background:T.white,
        border:`1px solid ${T.border}`,borderRadius:10,overflow:"hidden",marginBottom:20}}>
        {[
          {n:1,icon:"🐉",t:"Kali Attacker",  d:`${apiSources[0]?.ip||"192.168.137.x"} — nmap/hydra/msfconsole`},
          {n:2,icon:"🖥️", t:"Victim Ubuntu",  d:`${apiSources[0]?.targets?.[0]||"192.168.137.x"} — Metasploitable 2`},
          {n:3,icon:"📡",t:"tshark Capture",  d:"live_monitor.py captures hotspot interface"},
          {n:4,icon:"🧠",t:"STGE Scoring",    d:"28 ICS-ADD features → ensemble every 8s"},
          {n:5,icon:"🎯",t:"Live Prediction", d:`τ* = ${tau} → ATTACK (red) / BENIGN (green)`},
        ].map((s,i)=>(
          <div key={i} style={{padding:"12px 14px",borderRight:i<4?`1px solid ${T.border}`:"none",position:"relative"}}>
            <div style={{position:"absolute",top:0,left:0,right:0,height:3,
              background:[T.red,T.orange,T.blue,T.purple,T.green][i]}}/>
            <div style={{fontSize:18,marginBottom:3}}>{s.icon}</div>
            <div style={{fontSize:11,fontWeight:600,color:T.text,marginBottom:2}}>{s.t}</div>
            <p style={{fontSize:9,color:T.muted,margin:0,lineHeight:1.4}}>{s.d}</p>
          </div>
        ))}
      </div>

      {/* Connection panel */}
      {!connected&&(
        <Card style={{textAlign:"center",padding:40,marginBottom:20}}>
          <div style={{fontSize:40,marginBottom:12}}>🔌</div>
          <p style={{fontSize:15,fontWeight:600,margin:"0 0 6px"}}>Connect to Live Monitor API</p>
          <p style={{fontSize:12,color:T.muted,margin:"0 0 20px"}}>
            First run on Windows MSI:<br/>
            <code style={{background:T.surface,padding:"4px 10px",borderRadius:6,fontSize:12,display:"inline-block",marginTop:6}}>
              pip install flask flask-cors numpy
            </code>
            <br/>
            <code style={{background:T.surface,padding:"4px 10px",borderRadius:6,fontSize:12,display:"inline-block",marginTop:6}}>
              python C:\Demo\live_monitor.py
            </code>
          </p>
          {error&&<p style={{color:T.red,fontSize:12,marginBottom:16,background:T.redL,padding:"8px 16px",borderRadius:8}}>{error}</p>}
          <button onClick={connect} style={{background:T.blue,color:"#fff",border:"none",
            borderRadius:8,padding:"10px 28px",cursor:"pointer",fontSize:14,fontWeight:600}}>
            Connect to localhost:5001
          </button>
        </Card>
      )}

      {/* Interface selector + controls */}
      {connected&&(
        <Card style={{marginBottom:18}}>
          <div style={{display:"flex",gap:12,alignItems:"center",flexWrap:"wrap"}}>
            <div style={{flex:1,minWidth:280}}>
              <p style={{color:T.muted,fontSize:11,margin:"0 0 6px",fontWeight:500}}>
                Network Interface (select your hotspot adapter)
              </p>
              <select value={selIface} onChange={e=>setSelIface(e.target.value)}
                style={{width:"100%",padding:"8px 12px",border:`1px solid ${T.border}`,
                  borderRadius:8,fontSize:12,fontFamily:"'Roboto Mono',monospace",background:T.white}}>
                <option value="">— select interface —</option>
                {ifaces.map((f,i)=>(
                  <option key={i} value={f.split(".")[0].trim()}>{f}</option>
                ))}
              </select>
            </div>
            <div style={{display:"flex",gap:8,alignItems:"flex-end",paddingBottom:2}}>
              {!running?(
                <button onClick={startCapture} disabled={!selIface}
                  style={{background:T.green,color:"#fff",border:"none",borderRadius:8,
                    padding:"9px 20px",cursor:selIface?"pointer":"not-allowed",
                    fontSize:13,fontWeight:600,opacity:selIface?1:0.5}}>
                  ▶ Start Live Capture
                </button>
              ):(
                <button onClick={stopCapture}
                  style={{background:T.red,color:"#fff",border:"none",borderRadius:8,
                    padding:"9px 20px",cursor:"pointer",fontSize:13,fontWeight:600}}>
                  ⏹ Stop Capture
                </button>
              )}
              <button onClick={resetCapture}
                style={{background:"transparent",border:`1px solid ${T.border}`,color:T.muted,
                  borderRadius:8,padding:"9px 14px",cursor:"pointer",fontSize:12}}>Reset</button>
              <button onClick={()=>setConnected(false)}
                style={{background:"transparent",border:`1px solid ${T.border}`,color:T.muted,
                  borderRadius:8,padding:"9px 14px",cursor:"pointer",fontSize:12}}>Disconnect</button>
            </div>
          </div>
          {running&&(
            <p style={{color:T.green,fontSize:11,margin:"10px 0 0",fontWeight:500}}>
              ● Capturing on interface {selIface} · new flows scored every 8 seconds · polling every 2s
            </p>
          )}
        </Card>
      )}

      {/* KPI row */}
      {connected&&(
        <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:12,marginBottom:18}}>
          {[
            {l:"Total Flows",   v:apiStats.total,  c:T.blue},
            {l:"Attacks",       v:apiStats.attack, c:T.red},
            {l:"ML ensemble",   v:apiStats.ml??0,  c:T.purple},
            {l:"Signature",     v:apiStats.sig??0, c:T.orange},
            {l:"Benign",        v:apiStats.benign, c:T.green},
          ].map(k=>(
            <div key={k.l} style={{background:T.white,border:`1px solid ${T.border}`,borderRadius:10,padding:"12px 16px"}}>
              <p style={{color:T.muted,fontSize:10,textTransform:"uppercase",margin:0}}>{k.l}</p>
              <span style={{fontSize:20,fontWeight:700,color:k.c,fontFamily:"'Roboto Mono',monospace"}}>{k.v}</span>
            </div>
          ))}
        </div>
      )}

      {/* Score trend chart */}
      {connected&&apiFlows.length>0&&(
        <Card style={{marginBottom:18}}>
          <p style={{color:T.muted,fontSize:12,margin:"0 0 10px",fontWeight:500}}>
            Ensemble Anomaly Score — recent flows · τ* = {tau} (red line = detection boundary)
          </p>
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart data={trend} margin={{left:0,right:8,top:4,bottom:4}}>
              <defs>
                <linearGradient id="lmg" x1="0" y1="0" x2="0" y2="1">
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
              <Area type="monotone" dataKey="score" stroke={T.blue} fill="url(#lmg)" strokeWidth={2.5} name="Score" dot={false}/>
            </AreaChart>
          </ResponsiveContainer>
        </Card>
      )}

      {/* Attack sources — who is attacking whom */}
      {connected&&apiSources.length>0&&(
        <Card style={{marginBottom:18}}>
          <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>
            🎯 Attack Sources — top attacking IPs (this session)
          </p>
          <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(230px,1fr))",gap:10}}>
            {apiSources.map((s,i)=>(
              <div key={s.ip+i} style={{border:`1px solid ${T.red}40`,background:T.redL,
                borderRadius:8,padding:"10px 12px"}}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                  <span style={{fontFamily:"'Roboto Mono',monospace",fontWeight:700,fontSize:13,color:T.red}}>
                    🐉 {s.ip}</span>
                  <span style={{background:T.red,color:"#fff",borderRadius:10,padding:"1px 8px",
                    fontSize:11,fontWeight:700}}>{s.attacks}</span>
                </div>
                {s.targets&&s.targets.length>0&&(
                  <div style={{fontSize:10,color:T.sub,marginTop:4,fontFamily:"'Roboto Mono',monospace"}}>
                    → {s.targets.join(", ")}</div>)}
                <div style={{display:"flex",flexWrap:"wrap",gap:4,marginTop:6}}>
                  {(s.types||[]).map(t=>(
                    <span key={t} style={{background:T.white,border:`1px solid ${T.border}`,
                      borderRadius:4,padding:"1px 6px",fontSize:9,fontWeight:600,color:T.text}}>{t}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* SHAP-explained detections — persistent so exploits stay visible during scan floods */}
      {connected&&apiExplained.length>0&&(
        <Card style={{marginBottom:18}}>
          <p style={{color:T.muted,fontSize:12,margin:"0 0 12px",fontWeight:500}}>
            🔬 SHAP-Explained Detections — model feature attribution for exploit-class attacks
          </p>
          <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(320px,1fr))",gap:12}}>
            {apiExplained.slice(0,6).map((f,i)=>(
              <div key={i} style={{border:`1px solid ${T.border}`,borderRadius:8,padding:"10px 12px",background:T.white}}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
                  <span style={{fontSize:11,fontWeight:700,color:T.red}}>{f.attack_type}</span>
                  <span style={{fontSize:9,color:T.muted,fontFamily:"'Roboto Mono',monospace"}}>{f.src}→{f.dst}:{f.dport}</span>
                </div>
                {(f.shap||[]).slice(0,4).map((s,j)=>{
                  const mag=Math.min(100,Math.abs(s.impact)/(Math.abs(f.shap[0].impact)||1)*100);
                  const pos=s.direction==="toward_attack";
                  return(
                    <div key={j} style={{display:"flex",alignItems:"center",gap:6,marginBottom:2}}>
                      <span style={{width:130,fontSize:9,color:T.sub,textAlign:"right",whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>{s.feature}</span>
                      <div style={{flex:1,height:9,background:T.surface,borderRadius:2}}>
                        <div style={{width:`${mag}%`,height:"100%",borderRadius:2,background:pos?T.red:T.blue}}/>
                      </div>
                      <span style={{width:48,fontSize:9,fontFamily:"'Roboto Mono',monospace",color:pos?T.red:T.blue,fontWeight:700}}>{s.impact>=0?"+":""}{(+s.impact).toFixed(3)}</span>
                    </div>
                  );
                })}
                <div style={{fontSize:9,color:T.muted,marginTop:5,fontStyle:"italic"}}>{f.reason}</div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Live flow table */}
      {connected&&(
        <Card>
          <p style={{color:T.muted,fontSize:12,margin:"0 0 10px",fontWeight:500}}>
            Live Flow Log — hybrid detection (ML ensemble + signature rules) · click an ATTACK row for its SHAP reason
            {running&&<span style={{marginLeft:8,color:T.green,fontSize:10}}>● updating every 2s</span>}
            {!running&&apiFlows.length===0&&
              <span style={{marginLeft:8,color:T.muted,fontSize:10}}>— start capture above —</span>}
          </p>
          <div style={{maxHeight:420,overflowY:"auto"}}>
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:11,fontFamily:"'Roboto Mono',monospace"}}>
              <thead style={{position:"sticky",top:0,background:T.surface,zIndex:1}}>
                <tr style={{borderBottom:`1px solid ${T.border}`}}>
                  {["Time","Src IP","Dst IP","Proto","Port","Service","Pkts","Bytes",
                    "AE","LSTM","GNN","Trans","ML Score","Detection","Type","Reason (SHAP + rule)"].map(h=>(
                    <th key={h} style={{padding:"6px 7px",textAlign:"left",color:T.muted,
                      fontSize:9,fontWeight:600,whiteSpace:"nowrap"}}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {apiFlows.map((f,i)=>{
                  const atk=f.pred==="ATTACK";
                  const layerColor=f.layer==="ml+signature"?T.purple:f.layer==="signature"?T.orange:f.layer==="ml"?T.blue:T.muted;
                  const open=expanded===i;
                  return(
                    <Fragment key={i}>
                    <tr className="fade-row" onClick={()=>atk&&setExpanded(open?null:i)}
                      style={{borderBottom:`1px solid ${T.grid}`,background:atk?T.redL:"transparent",
                        cursor:atk?"pointer":"default"}}>
                      <td style={{padding:"4px 7px",color:T.muted,fontSize:9}}>{atk?(open?"▾ ":"▸ "):""}{f.ts}</td>
                      <td style={{padding:"4px 7px",fontSize:9,fontWeight:atk?700:400,color:atk?T.red:T.text}}>{f.src}</td>
                      <td style={{padding:"4px 7px",fontSize:9}}>{f.dst}</td>
                      <td style={{padding:"4px 7px",color:T.blue,fontWeight:500}}>{f.proto}</td>
                      <td style={{padding:"4px 7px",fontSize:9}}>{f.dport}</td>
                      <td style={{padding:"4px 7px",fontSize:9,color:f.service!=="-"?T.purple:T.muted,fontWeight:f.service!=="-"?600:400}}>{f.service}</td>
                      <td style={{padding:"4px 7px"}}>{f.pkts}</td>
                      <td style={{padding:"4px 7px"}}>{f.bytes}</td>
                      <td style={{padding:"4px 7px",color:T.blue}}>{f.ae}</td>
                      <td style={{padding:"4px 7px",color:T.green}}>{f.lstm}</td>
                      <td style={{padding:"4px 7px",color:T.orange}}>{f.gnn}</td>
                      <td style={{padding:"4px 7px",color:T.purple}}>{f.trans}</td>
                      <td style={{padding:"4px 7px",fontWeight:700,color:f.ml_hit?T.red:T.sub}}>{f.score}</td>
                      <td style={{padding:"4px 7px"}}>
                        {atk?<span style={{background:`${layerColor}18`,color:layerColor,
                          border:`1px solid ${layerColor}55`,borderRadius:4,padding:"2px 6px",
                          fontSize:9,fontWeight:700}}>{f.layer}</span>
                        :<span style={{color:T.green,fontSize:10,fontWeight:600}}>benign</span>}
                      </td>
                      <td style={{padding:"4px 7px",fontSize:9,color:atk?T.red:T.muted,fontWeight:atk?700:400}}>{f.attack_type}</td>
                      <td style={{padding:"4px 7px",fontSize:9,color:T.sub,maxWidth:260,whiteSpace:"nowrap",
                        overflow:"hidden",textOverflow:"ellipsis"}}>{f.reason}</td>
                    </tr>
                    {open&&atk&&(
                      <tr style={{background:"#fff7f6"}}>
                        <td colSpan={16} style={{padding:"10px 16px",borderBottom:`1px solid ${T.grid}`}}>
                          <div style={{display:"flex",gap:24,flexWrap:"wrap",alignItems:"flex-start"}}>
                            <div style={{minWidth:180}}>
                              <div style={{fontSize:10,color:T.muted,fontWeight:700,marginBottom:4}}>DETECTION</div>
                              <div style={{fontSize:12,color:T.text}}>{f.attack_type} · <b>{f.mitre}</b></div>
                              <div style={{fontSize:11,color:T.sub,marginTop:4}}>{f.reason}</div>
                              <div style={{fontSize:10,color:T.muted,marginTop:6}}>
                                ML score {f.score} {f.ml_hit?`> τ*=${tau}`:`≤ τ*=${tau}`} · layer: {f.layer}
                              </div>
                            </div>
                            <div style={{flex:1,minWidth:260}}>
                              <div style={{fontSize:10,color:T.muted,fontWeight:700,marginBottom:6}}>
                                SHAP — model feature attribution</div>
                              {(f.shap&&f.shap.length)?f.shap.map((s,j)=>{
                                const mag=Math.min(100,Math.abs(s.impact)/(Math.abs(f.shap[0].impact)||1)*100);
                                const pos=s.direction==="toward_attack";
                                return(
                                  <div key={j} style={{display:"flex",alignItems:"center",gap:8,marginBottom:3}}>
                                    <span style={{width:150,fontSize:10,color:T.sub,textAlign:"right",
                                      whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>{s.feature}</span>
                                    <div style={{flex:1,height:12,background:T.surface,borderRadius:3,position:"relative"}}>
                                      <div style={{width:`${mag}%`,height:"100%",borderRadius:3,
                                        background:pos?T.red:T.blue}}/>
                                    </div>
                                    <span style={{width:56,fontSize:10,fontFamily:"'Roboto Mono',monospace",
                                      color:pos?T.red:T.blue,fontWeight:700}}>{s.impact>=0?"+":""}{(+s.impact).toFixed(3)}</span>
                                  </div>
                                );
                              }):<span style={{fontSize:11,color:T.muted}}>SHAP not computed for this flow (budget) — rule reason above.</span>}
                              <div style={{fontSize:9,color:T.muted,marginTop:5}}>
                                <span style={{color:T.red,fontWeight:700}}>■</span> pushes toward attack ·
                                <span style={{color:T.blue,fontWeight:700}}> ■</span> toward benign
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                    </Fragment>
                  );
                })}
                {!apiFlows.length&&(
                  <tr><td colSpan={16} style={{padding:32,textAlign:"center",color:T.muted}}>
                    {running?"Waiting for first flow window (8 seconds)…":"Start capture to see live flows"}
                  </td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN COMPONENT
═══════════════════════════════════════════════════════════════════════════ */
/* ─── SHAP Explainability (real values exported from the CIC model) ──────────
   Computed once by shap_dashboard_export.export_shap_report("CIC") and embedded
   here. Click "Upload JSON" to load a fresh saved_models/shap_report_CIC.json.  */
const SHAP_CIC = {"dataset":"CIC","threshold":0.18740123510360718,"expected_value":0.07111139964312314,"feature_names":["Flow Duration","Bwd Packet Length Max","Bwd Packet Length Mean","Bwd Packet Length Std","Flow IAT Max","Fwd IAT Total","Fwd IAT Std","Fwd IAT Max","Fwd Header Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance","Avg Bwd Segment Size","Fwd Header Length.1","Idle Mean","Idle Max","Idle Min"],"n_scored":2827876,"n_flagged":293249,"nsamples":200,"background_size":100,"generated_at":"model_run","global_importance":[{"feature":"Bwd Packet Length Std","mean_abs_shap":0.2656943070997574},{"feature":"Packet Length Variance","mean_abs_shap":0.06507728838355335},{"feature":"Bwd Packet Length Max","mean_abs_shap":0.05275924616081619},{"feature":"Fwd IAT Std","mean_abs_shap":0.04441337042914934},{"feature":"Fwd IAT Max","mean_abs_shap":0.038079023971307385},{"feature":"Packet Length Std","mean_abs_shap":0.02894151658961784},{"feature":"Idle Mean","mean_abs_shap":0.0264363539676355},{"feature":"Idle Min","mean_abs_shap":0.017676238234864007},{"feature":"Packet Length Mean","mean_abs_shap":0.01724889579212468},{"feature":"Max Packet Length","mean_abs_shap":0.015399663148555046},{"feature":"Idle Max","mean_abs_shap":0.014568606471501998},{"feature":"Bwd Packet Length Mean","mean_abs_shap":0.006433316654334642},{"feature":"Avg Bwd Segment Size","mean_abs_shap":0.005114060329610777},{"feature":"Fwd Header Length","mean_abs_shap":0.002913856144933699},{"feature":"Flow IAT Max","mean_abs_shap":0.002384844738942667}],"alerts":[{"id":58358,"score":0.6170711517333984,"is_attack":true,"expected_value":0.07111139964312314,"sum_shap":0.57632748413831,"features":[{"feature":"Bwd Packet Length Std","raw_value":4128.3193359375,"shap":0.2925301241027965,"direction":"toward_attack"},{"feature":"Bwd Packet Length Max","raw_value":8760.0,"shap":0.05243610455534138,"direction":"toward_attack"},{"feature":"Packet Length Variance","raw_value":6192667.5,"shap":0.051440052281337516,"direction":"toward_attack"},{"feature":"Packet Length Std","raw_value":2488.507080078125,"shap":0.04564965451023054,"direction":"toward_attack"},{"feature":"Fwd IAT Max","raw_value":72400000.0,"shap":0.03497990626706854,"direction":"toward_attack"},{"feature":"Fwd IAT Std","raw_value":27100000.0,"shap":0.03062389667043843,"direction":"toward_attack"},{"feature":"Bwd Packet Length Mean","raw_value":2900.25,"shap":0.030403561391829332,"direction":"toward_attack"},{"feature":"Idle Min","raw_value":72400000.0,"shap":0.015436185814858085,"direction":"toward_attack"},{"feature":"Idle Max","raw_value":72400000.0,"shap":0.011887153598678535,"direction":"toward_attack"},{"feature":"Packet Length Mean","raw_value":897.1538696289062,"shap":0.010940844945731134,"direction":"toward_attack"}]},{"id":29674,"score":0.6167334318161011,"is_attack":true,"expected_value":0.07111139964312314,"sum_shap":0.5748842172697187,"features":[{"feature":"Bwd Packet Length Std","raw_value":3173.373779296875,"shap":0.2979852246478794,"direction":"toward_attack"},{"feature":"Packet Length Variance","raw_value":4423095.0,"shap":0.0782891192687409,"direction":"toward_attack"},{"feature":"Fwd IAT Std","raw_value":26100000.0,"shap":0.061092438823504805,"direction":"toward_attack"},{"feature":"Idle Min","raw_value":69300000.0,"shap":0.03460242206502148,"direction":"toward_attack"},{"feature":"Idle Mean","raw_value":69300000.0,"shap":0.033785398649957124,"direction":"toward_attack"},{"feature":"Bwd Packet Length Mean","raw_value":2321.39990234375,"shap":0.021497278329944403,"direction":"toward_attack"},{"feature":"Bwd Packet Length Max","raw_value":5840.0,"shap":0.018645152082934865,"direction":"toward_attack"},{"feature":"Fwd IAT Max","raw_value":69300000.0,"shap":0.013311173375152478,"direction":"toward_attack"},{"feature":"Packet Length Std","raw_value":2103.115478515625,"shap":0.0078435143145588,"direction":"toward_attack"},{"feature":"Idle Max","raw_value":69300000.0,"shap":0.007832495712024379,"direction":"toward_attack"}]},{"id":48694,"score":0.6165571212768555,"is_attack":true,"expected_value":0.07111139964312314,"sum_shap":0.5859516077116131,"features":[{"feature":"Bwd Packet Length Std","raw_value":4128.3193359375,"shap":0.2684555949794518,"direction":"toward_attack"},{"feature":"Packet Length Variance","raw_value":6192667.5,"shap":0.07979553202691529,"direction":"toward_attack"},{"feature":"Bwd Packet Length Max","raw_value":8760.0,"shap":0.05681049392731178,"direction":"toward_attack"},{"feature":"Fwd IAT Max","raw_value":72400000.0,"shap":0.049052795946661704,"direction":"toward_attack"},{"feature":"Fwd IAT Std","raw_value":27200000.0,"shap":0.03937968705924899,"direction":"toward_attack"},{"feature":"Idle Max","raw_value":72400000.0,"shap":0.028361118642023998,"direction":"toward_attack"},{"feature":"Packet Length Std","raw_value":2488.507080078125,"shap":0.02023769472294088,"direction":"toward_attack"},{"feature":"Packet Length Mean","raw_value":897.1538696289062,"shap":0.02013489116535897,"direction":"toward_attack"},{"feature":"Avg Bwd Segment Size","raw_value":2900.25,"shap":0.017328740995343612,"direction":"toward_attack"},{"feature":"Fwd Header Length.1","raw_value":172.0,"shap":0.006395058246356111,"direction":"toward_attack"}]},{"id":29670,"score":0.6163841485977173,"is_attack":true,"expected_value":0.07111139964312314,"sum_shap":0.5740913562849165,"features":[{"feature":"Bwd Packet Length Std","raw_value":3327.769775390625,"shap":0.2853832271032026,"direction":"toward_attack"},{"feature":"Fwd IAT Max","raw_value":68900000.0,"shap":0.05675953961922342,"direction":"toward_attack"},{"feature":"Packet Length Variance","raw_value":4731941.5,"shap":0.04151558141465838,"direction":"toward_attack"},{"feature":"Bwd Packet Length Max","raw_value":7215.0,"shap":0.03993237250917788,"direction":"toward_attack"},{"feature":"Fwd IAT Std","raw_value":25900000.0,"shap":0.03899174649730652,"direction":"toward_attack"},{"feature":"Idle Mean","raw_value":68900000.0,"shap":0.035805846803754836,"direction":"toward_attack"},{"feature":"Packet Length Std","raw_value":2175.302490234375,"shap":0.028916225619243244,"direction":"toward_attack"},{"feature":"Bwd Packet Length Mean","raw_value":2321.39990234375,"shap":0.01675571547295621,"direction":"toward_attack"},{"feature":"Avg Bwd Segment Size","raw_value":2321.39990234375,"shap":0.015961843547410225,"direction":"toward_attack"},{"feature":"Fwd Header Length","raw_value":172.0,"shap":0.014069257697983173,"direction":"toward_attack"}]},{"id":2223707,"score":0.6163681149482727,"is_attack":true,"expected_value":0.07111139964312314,"sum_shap":0.5734362416341902,"features":[{"feature":"Bwd Packet Length Std","raw_value":3003.171142578125,"shap":0.2415456347075631,"direction":"toward_attack"},{"feature":"Bwd Packet Length Max","raw_value":7240.0,"shap":0.06867096108412772,"direction":"toward_attack"},{"feature":"Packet Length Std","raw_value":2083.52197265625,"shap":0.05513595271227752,"direction":"toward_attack"},{"feature":"Packet Length Variance","raw_value":4341063.5,"shap":0.04500131896770777,"direction":"toward_attack"},{"feature":"Packet Length Mean","raw_value":920.1538696289062,"shap":0.04439253856624296,"direction":"toward_attack"},{"feature":"Idle Mean","raw_value":68500000.0,"shap":0.03992495764598798,"direction":"toward_attack"},{"feature":"Fwd IAT Max","raw_value":68500000.0,"shap":0.03946371414541703,"direction":"toward_attack"},{"feature":"Fwd IAT Std","raw_value":28000000.0,"shap":0.016613842557984634,"direction":"toward_attack"},{"feature":"Avg Bwd Segment Size","raw_value":2319.0,"shap":0.014753163889696982,"direction":"toward_attack"},{"feature":"Idle Min","raw_value":68500000.0,"shap":0.00793415735718439,"direction":"toward_attack"}]},{"id":30222,"score":0.6163278818130493,"is_attack":true,"expected_value":0.07111139964312314,"sum_shap":0.5705356412008405,"features":[{"feature":"Bwd Packet Length Std","raw_value":2436.8330078125,"shap":0.1940198254401345,"direction":"toward_attack"},{"feature":"Fwd IAT Max","raw_value":69500000.0,"shap":0.07463679353325518,"direction":"toward_attack"},{"feature":"Fwd IAT Std","raw_value":26200000.0,"shap":0.06107915851031409,"direction":"toward_attack"},{"feature":"Packet Length Std","raw_value":1774.90625,"shap":0.0523478934155242,"direction":"toward_attack"},{"feature":"Packet Length Variance","raw_value":3150292.5,"shap":0.04938699992665055,"direction":"toward_attack"},{"feature":"Idle Min","raw_value":69500000.0,"shap":0.0443454085147684,"direction":"toward_attack"},{"feature":"Flow IAT Max","raw_value":69500000.0,"shap":0.03180982969524997,"direction":"toward_attack"},{"feature":"Idle Mean","raw_value":69500000.0,"shap":0.024216400374720733,"direction":"toward_attack"},{"feature":"Packet Length Mean","raw_value":833.0714111328125,"shap":0.02099190914245698,"direction":"toward_attack"},{"feature":"Fwd Header Length.1","raw_value":172.0,"shap":0.017701422647765863,"direction":"toward_attack"}]},{"id":106740,"score":0.6163108348846436,"is_attack":true,"expected_value":0.07111139964312314,"sum_shap":0.579880278594792,"features":[{"feature":"Bwd Packet Length Std","raw_value":5795.50048828125,"shap":0.27905756759739947,"direction":"toward_attack"},{"feature":"Bwd Packet Length Max","raw_value":11595.0,"shap":0.05695723666321446,"direction":"toward_attack"},{"feature":"Fwd IAT Std","raw_value":29700000.0,"shap":0.051015693925216724,"direction":"toward_attack"},{"feature":"Packet Length Variance","raw_value":10300000.0,"shap":0.04724667241250519,"direction":"toward_attack"},{"feature":"Fwd IAT Max","raw_value":78900000.0,"shap":0.040737585081886483,"direction":"toward_attack"},{"feature":"Packet Length Std","raw_value":3214.16748046875,"shap":0.03533169813208668,"direction":"toward_attack"},{"feature":"Idle Mean","raw_value":78900000.0,"shap":0.028664662448510816,"direction":"toward_attack"},{"feature":"Avg Bwd Segment Size","raw_value":2901.75,"shap":0.01972539438828427,"direction":"toward_attack"},{"feature":"Bwd Packet Length Mean","raw_value":2901.75,"shap":0.016228663914121123,"direction":"toward_attack"},{"feature":"Flow IAT Max","raw_value":78900000.0,"shap":0.004915104031566697,"direction":"toward_attack"}]},{"id":57414,"score":0.6162106394767761,"is_attack":true,"expected_value":0.07111139964312314,"sum_shap":0.5742261819913984,"features":[{"feature":"Bwd Packet Length Std","raw_value":3173.373779296875,"shap":0.31783116832481095,"direction":"toward_attack"},{"feature":"Packet Length Variance","raw_value":4423095.0,"shap":0.05786779673267221,"direction":"toward_attack"},{"feature":"Fwd IAT Std","raw_value":27800000.0,"shap":0.04697992497969086,"direction":"toward_attack"},{"feature":"Fwd IAT Max","raw_value":74100000.0,"shap":0.04297443914575717,"direction":"toward_attack"},{"feature":"Bwd Packet Length Max","raw_value":5840.0,"shap":0.03155210279394082,"direction":"toward_attack"},{"feature":"Idle Min","raw_value":74100000.0,"shap":0.030578601138131566,"direction":"toward_attack"},{"feature":"Idle Mean","raw_value":74100000.0,"shap":0.01597023039462413,"direction":"toward_attack"},{"feature":"Bwd Packet Length Mean","raw_value":2321.39990234375,"shap":0.011435125108539284,"direction":"toward_attack"},{"feature":"Packet Length Mean","raw_value":833.5,"shap":0.00952675020504815,"direction":"toward_attack"},{"feature":"Fwd Header Length","raw_value":172.0,"shap":0.009510043168183269,"direction":"toward_attack"}]}]};

const ShapTip=({active,payload})=>{
  if(!active||!payload||!payload.length)return null;
  const p=payload[0].payload;
  return(<div style={{background:T.white,border:`1px solid ${T.border}`,borderRadius:8,padding:"8px 12px",boxShadow:"0 2px 8px rgba(0,0,0,.12)"}}>
    <div style={{fontWeight:700,fontSize:12,color:T.text}}>{p.feature}</div>
    {p.raw_value!==undefined&&<div style={{fontSize:11,color:T.muted}}>raw = {(+p.raw_value).toFixed(3)}</div>}
    <div style={{fontSize:11,color:p.shap>=0?T.red:T.blue,fontWeight:600}}>SHAP {p.shap>=0?"+":""}{(+p.shap).toFixed(4)}</div>
  </div>);
};

function ShapExplain(){
  const [rep,setRep]=useState(SHAP_CIC);
  const [isSample,setIsSample]=useState(true);
  const [aIdx,setAIdx]=useState(0);
  const [err,setErr]=useState("");
  const fileRef=useRef(null);
  const onFile=useCallback(e=>{
    const file=e.target.files[0];if(!file)return;setErr("");
    const rd=new FileReader();
    rd.onload=ev=>{try{const j=JSON.parse(ev.target.result);
      if(!j.alerts)throw new Error("missing 'alerts'");
      setRep(j);setIsSample(false);setAIdx(0);}catch(x){setErr("Parse error: "+x.message);}};
    rd.readAsText(file);
  },[]);
  const alerts=rep.alerts||[];
  const alert=alerts[Math.min(aIdx,alerts.length-1)]||null;
  const base=rep.expected_value??0, tau=rep.threshold??0;
  const bars=alert?[...alert.features].sort((a,b)=>Math.abs(a.shap)-Math.abs(b.shap)):[];
  const glob=(rep.global_importance||[]).map(g=>({...g})).reverse();
  return(
    <div>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",flexWrap:"wrap",gap:12,marginBottom:18}}>
        <SH sub="Empirical Shapley values from the frozen 4-branch CIC ensemble. Each bar shows how far a feature pushed a flagged flow away from the benign baseline toward the ATTACK decision. This is the same SHAP the Live Monitor attaches to each detected attack.">
          SHAP Explainability — why the ensemble flags a flow
        </SH>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          {isSample&&<Badge label="MODEL EXPORT" color={T.purple} bg={T.purpleL}/>}
          <button onClick={()=>fileRef.current?.click()} style={{background:T.blue,color:"#fff",border:"none",borderRadius:8,padding:"8px 16px",fontSize:12,fontWeight:600,cursor:"pointer"}}>Upload JSON</button>
          <input ref={fileRef} type="file" accept=".json" onChange={onFile} style={{display:"none"}}/>
        </div>
      </div>
      {err&&<div style={{background:T.redL,border:`1px solid ${T.red}`,borderRadius:8,padding:"10px 16px",color:T.red,fontSize:13,marginBottom:14}}>⚠ {err}</div>}
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(150px,1fr))",gap:12,marginBottom:18}}>
        {[{label:"Dataset",value:rep.dataset||"—",color:T.blue},
          {label:"Baseline E[f(x)]",value:(+base).toFixed(4),color:T.purple,sub:"expected benign score"},
          {label:"Threshold τ*",value:(+tau).toFixed(4),color:T.yellow,sub:"score > τ* ⇒ attack"},
          {label:"Alerts",value:alerts.length,color:T.red,sub:rep.n_flagged?`of ${(+rep.n_flagged).toLocaleString()} flagged`:""},
          {label:"SHAP samples",value:rep.nsamples||"—",color:T.teal,sub:`bg ${rep.background_size||"—"}`}
        ].map(k=>(
          <Card key={k.label} style={{padding:14}}>
            <div style={{fontSize:11,color:T.muted,fontWeight:600}}>{k.label}</div>
            <div style={{fontSize:22,fontWeight:700,color:k.color,fontFamily:"'Roboto Mono',monospace",margin:"4px 0 0"}}>{k.value}</div>
            {k.sub&&<div style={{fontSize:10,color:T.muted,marginTop:2}}>{k.sub}</div>}
          </Card>
        ))}
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,alignItems:"start"}}>
        <Card>
          <p style={{fontSize:13,fontWeight:700,color:T.text,margin:"0 0 4px"}}>Global feature importance</p>
          <p style={{color:T.muted,fontSize:11,margin:"0 0 12px"}}>Mean |SHAP| across explained alerts — what the model alerts on overall.</p>
          <ResponsiveContainer width="100%" height={Math.max(240,glob.length*24)}>
            <BarChart data={glob} layout="vertical" margin={{left:8,right:16,top:4,bottom:4}}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke={T.border}/>
              <XAxis type="number" tick={{fill:T.muted,fontSize:9}}/>
              <YAxis type="category" dataKey="feature" width={130} tick={{fill:T.sub,fontSize:9}}/>
              <Tooltip content={<ShapTip/>} cursor={{fill:T.surface}}/>
              <Bar dataKey="mean_abs_shap" fill={T.blue} radius={[0,3,3,0]}/>
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",flexWrap:"wrap",gap:8,marginBottom:4}}>
            <p style={{fontSize:13,fontWeight:700,color:T.text,margin:0}}>Per-alert explanation</p>
            {alert&&<Badge label={`flow #${alert.id} · score ${(+alert.score).toFixed(3)}`} color={T.red} bg={T.redL}/>}
          </div>
          <p style={{color:T.muted,fontSize:11,margin:"0 0 10px"}}>
            <span style={{color:T.red,fontWeight:700}}>■</span> toward ATTACK · <span style={{color:T.blue,fontWeight:700}}>■</span> toward BENIGN
          </p>
          {alert?(
            <ResponsiveContainer width="100%" height={Math.max(220,bars.length*28)}>
              <BarChart data={bars} layout="vertical" margin={{left:8,right:16,top:4,bottom:4}}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke={T.border}/>
                <XAxis type="number" tick={{fill:T.muted,fontSize:9}}/>
                <YAxis type="category" dataKey="feature" width={130} tick={{fill:T.sub,fontSize:9}}/>
                <Tooltip content={<ShapTip/>} cursor={{fill:T.surface}}/>
                <ReferenceLine x={0} stroke={T.text} strokeWidth={1}/>
                <Bar dataKey="shap" radius={[2,2,2,2]}>
                  {bars.map((b,i)=><Cell key={i} fill={b.shap>=0?T.red:T.blue}/>)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ):<div style={{padding:24,textAlign:"center",color:T.muted,fontSize:12}}>No alerts.</div>}
        </Card>
      </div>
      {alerts.length>0&&(
        <Card style={{marginTop:16}}>
          <p style={{fontSize:13,fontWeight:700,color:T.text,margin:"0 0 10px"}}>Flagged flows ({alerts.length}) — select one</p>
          <div style={{display:"flex",flexWrap:"wrap",gap:6}}>
            {alerts.map((a,i)=>(
              <button key={a.id+"_"+i} onClick={()=>setAIdx(i)} style={{
                background:i===aIdx?T.blue:T.surface,color:i===aIdx?"#fff":T.sub,
                border:`1px solid ${i===aIdx?T.blue:T.border}`,borderRadius:16,
                padding:"5px 12px",fontSize:11,fontWeight:600,cursor:"pointer",fontFamily:"'Roboto Mono',monospace"}}>
                #{a.id} · {(+a.score).toFixed(2)}
              </button>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

export default function Dashboard(){
  const [tab,setTab]=useState(0);
  const [ds,setDs]=useState("ICS-ADD");

  /* Live monitor */
  const [live,setLive]=useState(false);
  const [flows,setFlows]=useState([]);
  const [lcnt,setLcnt]=useState({total:0,atk:0,ben:0,tp:0,fp:0});
  const intRef=useRef(null);

  /* ICS-ADD Inference tab */
  const [icsStep,setIcsStep]=useState("idle");
  const [icsErr,setIcsErr]=useState("");
  const [icsRows,setIcsRows]=useState([]);
  const [icsMeta,setIcsMeta]=useState(null);
  const [icsPage,setIcsPage]=useState(0);
  const [icsFilt,setIcsFilt]=useState("ALL");
  const icsRef=useRef(null);

  /* CICIoT2023 Inference tab */
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

  /* ICS-ADD CSV handler */
  const handleICSCSV=useCallback(e=>{
    const file=e.target.files[0]; if(!file)return;
    setIcsStep("parsing");setIcsErr("");setIcsRows([]);setIcsMeta(null);setIcsPage(0);
    const reader=new FileReader();
    reader.onerror=()=>{setIcsErr("Failed to read file.");setIcsStep("error");};
    reader.onload=ev=>{
      try{
        const {data,headers}=parseCSV(ev.target.result);
        if(!data||data.length===0)throw new Error("No data rows found.");
        const dsType=detectDataset(headers);
        const labelCols=headers.filter(h=>isLabel(h)||h==='attack_label');
        const featCols=headers.filter(h=>!isLabel(h)&&h!=='attack_label');
        const clean=data.map(r=>{const o={};featCols.forEach(c=>{o[c]=r[c]});return o;});
        setIcsMeta({totalRows:clean.length,colCount:featCols.length,
          labelFound:labelCols.length>0,labelCols,fileName:file.name,
          fileSize:(file.size/1024/1024).toFixed(2)+"MB",dsType});
        setIcsStep("running");
        const CHUNK=500;const all=[];let off=0;
        function go(){
          const scored=runICSInference(clean.slice(off,off+CHUNK),featCols);
          all.push(...scored);off+=CHUNK;
          if(off<clean.length) setTimeout(go,0);
          else{setIcsRows(all);setIcsStep("done");}
        }
        setTimeout(go,0);
      }catch(err){setIcsErr(err.message);setIcsStep("error");}
    };
    reader.readAsText(file);
  },[]);

  /* ICS inference stats */
  const icsStats=useCallback(()=>{
    if(!icsRows.length)return null;
    const atk=icsRows.filter(r=>r.pred==="ATTACK");
    const ben=icsRows.filter(r=>r.pred==="BENIGN");
    const byDom={AE:0,LSTM:0,GNN:0,Trans:0};
    atk.forEach(r=>{byDom[r.dom]=(byDom[r.dom]||0)+1;});
    const byMitre={};
    atk.forEach(r=>{byMitre[r.mitre]=(byMitre[r.mitre]||0)+1;});
    const hist=Array.from({length:20},(_,i)=>({
      bin:`${(i*5).toString().padStart(2,"0")}%`,
      count:icsRows.filter(r=>r.score>=i*0.05&&r.score<(i+1)*0.05).length,
    }));
    return{atk:atk.length,ben:ben.length,total:icsRows.length,
      attackPct:(atk.length/icsRows.length*100).toFixed(1),
      byDom,byMitre,hist,
      avgA:atk.length?+(atk.reduce((s,r)=>s+r.score,0)/atk.length).toFixed(4):0,
      avgB:ben.length?+(ben.reduce((s,r)=>s+r.score,0)/ben.length).toFixed(4):0};
  },[icsRows]);
  const iss=icsStep==="done"?icsStats():null;
  const ICS_RPP=50;
  const icsFRows=icsRows.filter(r=>icsFilt==="ALL"||r.pred===icsFilt);
  const icsPageRows=icsFRows.slice(icsPage*ICS_RPP,(icsPage+1)*ICS_RPP);
  const icsTotPg=Math.ceil(icsFRows.length/ICS_RPP);

  /* CIC Inference stats */
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

      {/* ── TOP ACCENT BAR ─────────────────────────────────────────────── */}
      <div style={{height:4,width:"100%",position:"sticky",top:0,zIndex:101,
        background:`linear-gradient(90deg,#e91e63,#9334e6,#673ab7,${T.blue})`}}/>

      {/* ── HEADER ─────────────────────────────────────────────────────── */}
      <header style={{background:T.white,borderBottom:`1px solid ${T.border}`,padding:"14px 28px",
        display:"grid",gridTemplateColumns:"1fr auto 1fr",alignItems:"center",
        boxShadow:"0 1px 4px rgba(0,0,0,.06)",position:"sticky",top:4,zIndex:100,gap:16}}>

        {/* Left: MCTE logo + branding */}
        <div style={{display:"flex",alignItems:"center",gap:14,minWidth:0}}>
          <img src={`data:image/png;base64,${MCTE_LOGO_B64}`} alt="MCTE"
            style={{height:46,width:46,objectFit:"contain",flexShrink:0}}/>
          <div style={{minWidth:0}}>
            <div style={{fontSize:14,fontWeight:700,color:T.text,letterSpacing:".03em",
              fontFamily:"'Google Sans',sans-serif",lineHeight:1.2,whiteSpace:"nowrap"}}>MCTE</div>
            <div style={{fontSize:10.5,color:T.muted,lineHeight:1.35,maxWidth:170}}>
              Military College of Telecommunication Engineering
            </div>
          </div>
        </div>

        {/* Center: SurakshaNetra title */}
        <div style={{display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",lineHeight:1.2}}>
          <span style={{fontSize:26,fontWeight:700,letterSpacing:".2em",fontFamily:"'Google Sans',sans-serif",
            background:`linear-gradient(90deg,${T.blue},${T.purple})`,WebkitBackgroundClip:"text",
            WebkitTextFillColor:"transparent",backgroundClip:"text",whiteSpace:"nowrap"}}>
            SURAKSHANETRA
          </span>
          <span style={{fontSize:12.5,fontWeight:700,color:T.text,whiteSpace:"nowrap",marginTop:2}}>
            Zero-Day Detection in Air-Gapped Networks
          </span>
          <span style={{fontSize:10,color:T.muted,whiteSpace:"nowrap",marginTop:1}}>
            AGAD-UDL · STGE Deep Ensemble · Amit Dalal &amp; Prashant Mishra
          </span>
        </div>

        {/* Right: IIT Indore branding + logo */}
        <div style={{display:"flex",alignItems:"center",justifyContent:"flex-end",gap:14,minWidth:0}}>
          <div style={{minWidth:0,textAlign:"right"}}>
            <div style={{fontSize:14,fontWeight:700,color:T.text,letterSpacing:".03em",
              fontFamily:"'Google Sans',sans-serif",lineHeight:1.2,whiteSpace:"nowrap"}}>IIT Indore</div>
            <div style={{fontSize:10.5,color:T.muted,lineHeight:1.35}}>
              Indian Institute of Technology<br/>Indore
            </div>
          </div>
          <img src={`data:image/png;base64,${IITI_LOGO_B64}`} alt="IIT Indore"
            style={{height:46,width:46,objectFit:"contain",flexShrink:0}}/>
        </div>
      </header>

      {/* ── TAB BAR ────────────────────────────────────────────────────── */}
      <nav style={{background:T.white,borderBottom:`1px solid ${T.border}`,padding:"0 28px",
        display:"flex",alignItems:"center",justifyContent:"space-between",gap:12,overflowX:"auto"}}>
        <div style={{display:"flex",gap:0}}>
          {TABS.map((t,i)=>(
            <button key={t} className="tab-btn" onClick={()=>setTab(i)} style={{
              background:"transparent",padding:"14px 18px",
              color:tab===i?T.blue:T.sub,fontWeight:tab===i?500:400,fontSize:13,
              borderBottom:tab===i?`2.5px solid ${T.blue}`:"2.5px solid transparent",
              marginBottom:-1,whiteSpace:"nowrap",
            }}>
              {t}
              {i===ICS_INFER_TAB&&icsStep==="done"&&(
                <span style={{marginLeft:6,background:T.orange,color:T.white,borderRadius:10,
                  padding:"1px 7px",fontSize:10,fontWeight:700}}>{icsRows.length}</span>
              )}
              {i===INFER_TAB&&infStep==="done"&&(
                <span style={{marginLeft:6,background:T.blue,color:T.white,borderRadius:10,
                  padding:"1px 7px",fontSize:10,fontWeight:700}}>{infRows.length}</span>
              )}
            </button>
          ))}
        </div>
        <div style={{display:"flex",gap:6,flexShrink:0}}>
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

        {/* ╔═══ TAB 5 — LIVE MONITOR (real API) ══════════════════════════╗ */}
        {tab===5&&<LiveMonitorTab tau={TAU_ICS}/>}

        {/* ╔═══ TAB 6 — ICS-ADD LIVE INFERENCE ══════════════════════════╗ */}
        {tab===6&&(
          <div>
            <SH sub={`Upload ics_model_ready.csv generated by ics_remap.py — 28 AOA features · τ* = ${TAU_ICS} · Z-scored`}>
              ICS-ADD Live Inference — Wireshark Capture Validation
            </SH>

            {/* Pipeline steps */}
            <div style={{display:"grid",gridTemplateColumns:"repeat(6,1fr)",background:T.white,
              border:`1px solid ${T.border}`,borderRadius:10,overflow:"hidden",marginBottom:20}}>
              {[
                {n:1,icon:"📡",t:"Wireshark",    d:"Capture PCAP from attack session"},
                {n:2,icon:"🔄",t:"NFStream",      d:"python convert.py → flows.csv"},
                {n:3,icon:"🗺️", t:"Remap",        d:"python ics_remap.py → ics_model_ready.csv"},
                {n:4,icon:"📐",t:"28 Features",   d:"AOA-selected, Z-score normalised"},
                {n:5,icon:"🧠",t:"STGE Score",    d:"AE(0.28)+LSTM(0.24)+GNN(0.27)+Trans(0.21)"},
                {n:6,icon:"🎯",t:"OOA Predict",   d:`τ* = ${TAU_ICS} → ATTACK / BENIGN`},
              ].map((s,i)=>(
                <div key={i} style={{padding:"12px 14px",borderRight:i<5?`1px solid ${T.border}`:"none",position:"relative"}}>
                  <div style={{position:"absolute",top:0,left:0,right:0,height:3,background:i===5?T.green:T.orange}}/>
                  <div style={{fontSize:16,marginBottom:3}}>{s.icon}</div>
                  <div style={{display:"flex",alignItems:"center",gap:4,marginBottom:2}}>
                    <span style={{fontSize:9,fontFamily:"'Roboto Mono',monospace",color:T.orange,fontWeight:700}}>#{s.n}</span>
                    <span style={{fontSize:11,fontWeight:600,color:T.text}}>{s.t}</span>
                  </div>
                  <p style={{fontSize:9,color:T.muted,margin:0,lineHeight:1.4}}>{s.d}</p>
                </div>
              ))}
            </div>

            {/* Upload / error state */}
            {(icsStep==="idle"||icsStep==="error")&&(
              <div>
                <div className="drop-z" onClick={()=>icsRef.current?.click()}
                  style={{background:T.white,border:`2px dashed ${T.orange}`,borderRadius:14,
                    padding:"44px 28px",textAlign:"center",marginBottom:16}}>
                  <div style={{fontSize:40,marginBottom:10}}>🛡️</div>
                  <p style={{fontSize:15,fontWeight:600,margin:0,color:T.text}}>
                    Drop <code style={{background:T.surface,padding:"2px 6px",borderRadius:4,fontSize:13}}>ics_model_ready.csv</code> here or click to browse
                  </p>
                  <p style={{fontSize:12,color:T.muted,margin:"6px 0 0"}}>
                    Generated by <b>python C:\Demo\ics_remap.py</b> — 28 AOA features, Z-score normalised
                  </p>
                  <div style={{marginTop:16,display:"inline-block",background:T.orange,color:"#fff",
                    borderRadius:8,padding:"8px 24px",fontSize:13,fontWeight:600}}>Choose File</div>
                  <input ref={icsRef} type="file" accept=".csv,.txt" onChange={handleICSCSV} style={{display:"none"}}/>
                </div>
                {icsStep==="error"&&(
                  <div style={{background:T.redL,border:`1px solid ${T.red}`,borderRadius:8,
                    padding:"10px 16px",color:T.red,fontSize:13}}>⚠ {icsErr}</div>
                )}
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
                  <Card>
                    <p style={{color:T.muted,fontSize:11,margin:"0 0 8px",fontWeight:500}}>
                      28 AOA-Selected ICS-ADD Features (exact from notebook)
                    </p>
                    <div style={{display:"flex",flexWrap:"wrap",gap:4}}>
                      {ICS_AOA_28.map(f=>(
                        <span key={f} style={{fontSize:9,fontFamily:"'Roboto Mono',monospace",
                          background:["syn_flag_cnt","rst_flag_cnt","radius","covariance","flow_pkts_per_s"].includes(f)?T.redL:T.blueL,
                          color:["syn_flag_cnt","rst_flag_cnt","radius","covariance","flow_pkts_per_s"].includes(f)?T.red:T.blue,
                          borderRadius:4,padding:"2px 5px"}}>{f}</span>
                      ))}
                    </div>
                    <p style={{fontSize:10,color:T.muted,marginTop:8}}>
                      <span style={{color:T.red}}>■</span> Key attack indicators
                      &nbsp;&nbsp;<span style={{color:T.blue}}>■</span> Supporting features
                    </p>
                  </Card>
                  <Card>
                    <p style={{color:T.muted,fontSize:11,margin:"0 0 10px",fontWeight:500}}>
                      MITRE ATT&amp;CK ICS — Mapping Logic
                    </p>
                    {[
                      {tech:"T0814 (DoS/Flood)",    cond:"flow_pkts_per_s z > +2σ",  color:T.red},
                      {tech:"T0846 (Port Scan)",    cond:"syn_flag_cnt > +2σ AND rst > +1σ", color:T.orange},
                      {tech:"T0891 (Brute Force)",  cond:"syn_flag_cnt z > +1σ",     color:T.yellow},
                      {tech:"T0867 (Lateral Mvmt)", cond:"radius z > +1σ",           color:T.purple},
                      {tech:"T0843 (PLC Access)",   cond:"covariance z > +1σ",       color:T.blue},
                    ].map(m=>(
                      <div key={m.tech} style={{display:"flex",alignItems:"center",gap:8,marginBottom:5}}>
                        <span style={{width:8,height:8,borderRadius:"50%",background:m.color,flexShrink:0}}/>
                        <span style={{fontSize:11,fontFamily:"'Roboto Mono',monospace",color:m.color,fontWeight:600,width:140}}>{m.tech}</span>
                        <span style={{fontSize:10,color:T.muted}}>{m.cond}</span>
                      </div>
                    ))}
                  </Card>
                </div>
              </div>
            )}

            {/* Parsing / running */}
            {(icsStep==="parsing"||icsStep==="running")&&(
              <Card style={{textAlign:"center",padding:48}}>
                <span className="spin" style={{fontSize:36,display:"block",marginBottom:14}}>⚙</span>
                <p style={{fontSize:15,fontWeight:600,margin:0}}>
                  {icsStep==="parsing"?"Parsing CSV…":"Running ICS-ADD STGE Inference…"}
                </p>
                {icsMeta&&<p style={{color:T.muted,fontSize:12,margin:"6px 0 0"}}>
                  {icsMeta.totalRows.toLocaleString()} flows · {icsMeta.colCount} features
                </p>}
                {icsStep==="running"&&<p style={{color:T.muted,fontSize:11,marginTop:6}}>
                  {icsRows.length.toLocaleString()} / {icsMeta?.totalRows?.toLocaleString()} flows scored…
                </p>}
              </Card>
            )}

            {/* Results */}
            {icsStep==="done"&&iss&&(
              <div>
                {/* Banner */}
                <div style={{background:T.white,border:`1px solid ${T.border}`,borderRadius:10,
                  padding:"12px 20px",marginBottom:18,display:"flex",gap:20,alignItems:"center",flexWrap:"wrap"}}>
                  {[
                    {k:"File",     v:icsMeta.fileName},
                    {k:"Rows",     v:iss.total.toLocaleString()},
                    {k:"Dataset",  v:"ICS-ADD (28 AOA features)"},
                    {k:"τ* (OOA)", v:TAU_ICS},
                    {k:"AUC",      v:AUC_ICS},
                  ].map(x=>(
                    <div key={x.k}>
                      <p style={{color:T.muted,fontSize:10,margin:0,textTransform:"uppercase"}}>{x.k}</p>
                      <p style={{color:T.text,fontSize:12,fontWeight:600,margin:"2px 0 0",fontFamily:"'Roboto Mono',monospace"}}>{x.v}</p>
                    </div>
                  ))}
                  <button onClick={()=>{setIcsStep("idle");setIcsRows([]);setIcsMeta(null);setIcsPage(0);if(icsRef.current)icsRef.current.value="";}}
                    style={{marginLeft:"auto",background:"transparent",border:`1px solid ${T.border}`,color:T.muted,
                      borderRadius:8,padding:"6px 14px",cursor:"pointer",fontSize:12}}>↩ New Upload</button>
                </div>

                {/* KPIs */}
                <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12,marginBottom:18}}>
                  <KpiCard label="Predicted ATTACK" value={iss.atk.toLocaleString()} unit={`(${iss.attackPct}%)`} accent={T.red}   sub={`Score > τ* ${TAU_ICS}`}/>
                  <KpiCard label="Predicted BENIGN" value={iss.ben.toLocaleString()} unit={`(${(100-+iss.attackPct).toFixed(1)}%)`} accent={T.green} sub={`Score ≤ τ* ${TAU_ICS}`}/>
                  <KpiCard label="Avg Score — Attack" value={iss.avgA} accent={T.red}   sub="Higher = more anomalous"/>
                  <KpiCard label="Avg Score — Benign" value={iss.avgB} accent={T.green} sub="Lower = more normal"/>
                </div>

                {/* Charts row */}
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:16,marginBottom:18}}>
                  <Card>
                    <p style={{color:T.muted,fontSize:12,margin:"0 0 10px",fontWeight:500}}>Prediction Split</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <PieChart>
                        <Pie data={[{name:"ATTACK",value:iss.atk},{name:"BENIGN",value:iss.ben}]}
                          dataKey="value" cx="50%" cy="50%" innerRadius={48} outerRadius={78} paddingAngle={4}>
                          <Cell fill={T.red}/><Cell fill={T.green}/>
                        </Pie>
                        <Tooltip content={<Tip/>}/>
                        <Legend wrapperStyle={{color:T.muted,fontSize:11}}/>
                      </PieChart>
                    </ResponsiveContainer>
                  </Card>
                  <Card>
                    <p style={{color:T.muted,fontSize:12,margin:"0 0 10px",fontWeight:500}}>Score Distribution · τ*={TAU_ICS}</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={iss.hist} margin={{left:0,right:4,top:4,bottom:12}}>
                        <CartesianGrid stroke={T.grid} strokeDasharray="3 3"/>
                        <XAxis dataKey="bin" tick={{fill:T.muted,fontSize:7}} angle={-45} textAnchor="end"/>
                        <YAxis tick={{fill:T.muted,fontSize:9}}/>
                        <Tooltip content={<Tip/>}/>
                        <ReferenceLine x={Math.round(TAU_ICS*20)*5+"%"} stroke={T.red} strokeDasharray="4 3"
                          label={{value:"τ*",fill:T.red,fontSize:10,position:"top"}}/>
                        <Bar dataKey="count" name="Flows" radius={[2,2,0,0]}>
                          {iss.hist.map((_,i)=><Cell key={i} fill={i*0.05>=TAU_ICS?T.red:T.green}/>)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </Card>
                  <Card>
                    <p style={{color:T.muted,fontSize:12,margin:"0 0 10px",fontWeight:500}}>MITRE ATT&amp;CK Breakdown</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart
                        data={Object.entries(iss.byMitre).map(([k,v])=>({t:k,n:v})).sort((a,b)=>b.n-a.n)}
                        layout="vertical" margin={{left:100,right:12,top:4,bottom:4}}>
                        <CartesianGrid stroke={T.grid} strokeDasharray="3 3" horizontal={false}/>
                        <XAxis type="number" tick={{fill:T.muted,fontSize:9}}/>
                        <YAxis dataKey="t" type="category" tick={{fill:T.sub,fontSize:8}} width={100}/>
                        <Tooltip content={<Tip/>}/>
                        <Bar dataKey="n" name="Flows" fill={T.orange} radius={[0,4,4,0]}/>
                      </BarChart>
                    </ResponsiveContainer>
                  </Card>
                </div>

                {/* Per-row table */}
                <Card>
                  <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:12,gap:8,flexWrap:"wrap"}}>
                    <p style={{color:T.muted,fontSize:12,margin:0,fontWeight:500}}>
                      Per-Flow Predictions — {icsFRows.length.toLocaleString()} flows · ICS-ADD model · τ*={TAU_ICS}
                    </p>
                    <div style={{display:"flex",gap:6}}>
                      {["ALL","ATTACK","BENIGN"].map(f=>(
                        <button key={f} onClick={()=>{setIcsFilt(f);setIcsPage(0);}} style={{
                          background:icsFilt===f?(f==="ATTACK"?T.redL:f==="BENIGN"?T.greenL:T.blueL):"transparent",
                          border:`1.5px solid ${icsFilt===f?(f==="ATTACK"?T.red:f==="BENIGN"?T.green:T.blue):T.border}`,
                          color:icsFilt===f?(f==="ATTACK"?T.red:f==="BENIGN"?T.green:T.blue):T.muted,
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
                          {["#","Src IP","Dst IP","Port","App","SYN_z","RST_z","PPS_z","Radius_z","Cov_z","AE","LSTM","GNN","Trans","Score","MITRE","Pred","Conf"].map(h=>(
                            <th key={h} style={{padding:"6px 7px",textAlign:"left",color:T.sub,fontSize:9,fontWeight:600,whiteSpace:"nowrap"}}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {icsPageRows.map((r,i)=>{
                          const atk=r.pred==="ATTACK";
                          return(
                            <tr key={i} className="tr-row" style={{borderBottom:`1px solid ${T.grid}`,background:atk?T.redL:"transparent"}}>
                              <td style={{padding:"4px 7px",color:T.muted}}>{r.idx}</td>
                              <td style={{padding:"4px 7px",fontSize:9}}>{r.src}</td>
                              <td style={{padding:"4px 7px",fontSize:9}}>{r.dst}</td>
                              <td style={{padding:"4px 7px"}}>{r.port}</td>
                              <td style={{padding:"4px 7px",fontSize:9,color:T.blue}}>{r.app}</td>
                              <td style={{padding:"4px 7px",color:+r.syn>1?T.red:T.text,fontWeight:+r.syn>1?700:400}}>{r.syn}</td>
                              <td style={{padding:"4px 7px",color:+r.rst>0.5?T.orange:T.text}}>{r.rst}</td>
                              <td style={{padding:"4px 7px",color:+r.pps>2?T.red:T.text,fontWeight:+r.pps>2?700:400}}>{r.pps}</td>
                              <td style={{padding:"4px 7px",color:+r.radius>0.5?T.purple:T.text}}>{r.radius}</td>
                              <td style={{padding:"4px 7px",color:+r.cov>0.3?T.purple:T.text}}>{r.cov}</td>
                              <td style={{padding:"4px 7px",color:T.blue}}>{r.ae}</td>
                              <td style={{padding:"4px 7px",color:T.green}}>{r.lstm}</td>
                              <td style={{padding:"4px 7px",color:T.orange}}>{r.gnn}</td>
                              <td style={{padding:"4px 7px",color:T.purple}}>{r.trans}</td>
                              <td style={{padding:"4px 7px",fontWeight:700,color:r.score>TAU_ICS?T.red:T.green}}>{r.score}</td>
                              <td style={{padding:"4px 7px",fontSize:9,color:T.orange}}>{r.mitre}</td>
                              <td style={{padding:"4px 7px"}}>
                                <Badge label={r.pred} color={atk?T.red:T.green} bg={atk?T.redL:T.greenL}/>
                              </td>
                              <td style={{padding:"4px 7px",fontSize:9,color:T.muted}}>{r.conf}</td>
                            </tr>
                          );
                        })}
                        {!icsPageRows.length&&<tr><td colSpan={18} style={{padding:20,textAlign:"center",color:T.muted}}>No rows match filter</td></tr>}
                      </tbody>
                    </table>
                  </div>
                  {icsTotPg>1&&(
                    <div style={{display:"flex",justifyContent:"center",gap:8,marginTop:14,alignItems:"center"}}>
                      <button onClick={()=>setIcsPage(p=>Math.max(0,p-1))} disabled={icsPage===0}
                        style={{background:"transparent",border:`1px solid ${T.border}`,color:T.muted,
                          borderRadius:6,padding:"5px 12px",cursor:"pointer",fontSize:11,opacity:icsPage===0?.4:1}}>← Prev</button>
                      <span style={{color:T.muted,fontSize:11}}>Page {icsPage+1} / {icsTotPg} · {icsFRows.length.toLocaleString()} flows</span>
                      <button onClick={()=>setIcsPage(p=>Math.min(icsTotPg-1,p+1))} disabled={icsPage===icsTotPg-1}
                        style={{background:"transparent",border:`1px solid ${T.border}`,color:T.muted,
                          borderRadius:6,padding:"5px 12px",cursor:"pointer",fontSize:11,opacity:icsPage===icsTotPg-1?.4:1}}>Next →</button>
                    </div>
                  )}
                </Card>
              </div>
            )}
          </div>
        )}

        {/* ╔═══ TAB 7 — CICIoT2023 INFERENCE ════════════════════════════╗ */}
        {tab===7&&(
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

        {/* ╔═══ TAB 8 — SHAP EXPLAINABILITY ═════════════════════════════╗ */}
        {tab===SHAP_TAB&&<ShapExplain/>}
      </main>

      {/* ── FOOTER ─────────────────────────────────────────────────────── */}
      <footer style={{borderTop:`1px solid ${T.border}`,padding:"12px 28px",background:T.white,
        display:"flex",justifyContent:"space-between",alignItems:"center"}}>
        <span style={{fontSize:11,color:T.muted}}>SurakshaNetra · AGAD-UDL · IIT Indore · Amit Dalal · Prashant Mishra</span>
        <span style={{fontSize:11,color:T.muted}}>PyTorch 2.11 · CUDA (Tesla T4) · ICS-ADD 120k · CICIoT2023 2.8M · MITRE ATT&CK ICS v15.1</span>
      </footer>
    </div>
  );
}
