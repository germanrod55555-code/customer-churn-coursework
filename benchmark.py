"""Нагрузка на пакетное прогнозирование. Повторы не используются для обучения."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from churn import FEATURES, clean_names, predict_file

def make_load(path, rows):
    base=clean_names(pd.read_csv(ROOT/'data/customer_churn.csv'))[FEATURES]
    with pq.ParquetWriter(path,pa.Table.from_pandas(base,preserve_index=False).schema,compression='snappy') as writer:
        for start in range(0,rows,25000):
            idx=np.arange(start,min(rows,start+25000))%len(base)
            table=pa.Table.from_pandas(base.iloc[idx].reset_index(drop=True),preserve_index=False)
            writer.write_table(table)

def worker(source,target,batch):
    process=psutil.Process(os.getpid()); initial=process.memory_info().rss
    peak=[initial]; done=threading.Event()
    def monitor():
        while not done.wait(.01):peak[0]=max(peak[0],process.memory_info().rss)
    thread=threading.Thread(target=monitor,daemon=True); thread.start()
    start=time.perf_counter()
    rows=predict_file(source,target,ROOT/'results/model.joblib',batch)
    elapsed=time.perf_counter()-start
    peak[0]=max(peak[0],process.memory_info().rss); done.set(); thread.join()
    # Полноту проверяем после остановки измерений, чтением выходного CSV порциями.
    count=sum(len(c) for c in pd.read_csv(target,chunksize=25000))
    assert count==rows
    print(json.dumps({'rows':rows,'batch':batch,'seconds':elapsed,'rows_per_second':rows/elapsed,
                      'peak_rss_mib':peak[0]/2**20,'initial_rss_mib':initial/2**20,
                      'delta_rss_mib':(peak[0]-initial)/2**20,'verified_rows':count}))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--worker',nargs=3)
    args=parser.parse_args()
    if args.worker:return worker(Path(args.worker[0]),Path(args.worker[1]),int(args.worker[2]))
    records=[]
    with tempfile.TemporaryDirectory(prefix='churn_load_') as td:
        temp=Path(td)
        for rows in (10000,100000,500000):
            source=temp/f'load_{rows}.parquet'; make_load(source,rows)
            for batch in (5000,25000):
                for repeat in range(3):
                    target=temp/f'out_{rows}_{batch}_{repeat}.csv'
                    result=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).resolve()),
                        '--worker',str(source),str(target),str(batch)],capture_output=True,text=True,check=True)
                    rec=json.loads(result.stdout);rec.update(repeat=repeat+1,parquet_bytes=source.stat().st_size)
                    records.append(rec); target.unlink()
                    print(f'{rows} строк, пакет {batch}, повтор {repeat+1}: {rec["seconds"]:.2f} с',flush=True)
            source.unlink()
    frame=pd.DataFrame(records); frame.to_csv(ROOT/'results/load_measurements.csv',index=False)
    frame.groupby(['rows','batch'],as_index=False).agg(seconds=('seconds','median'),
        rows_per_second=('rows_per_second','median'),peak_rss_mib=('peak_rss_mib','median'),
        delta_rss_mib=('delta_rss_mib','median')).to_csv(ROOT/'results/load_summary.csv',index=False)

if __name__=='__main__':main()
