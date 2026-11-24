import sys
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from churn import (FEATURES, TARGET, NUMERIC, load_data, split_data, preprocessing,
                   validate_features, clean_names, select_threshold, predict_frame, predict_file)

class ChurnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.audit=load_data(ROOT/'data/customer_churn.csv')
        cls.splits=split_data(cls.data)
        cls.train=cls.data.iloc[cls.splits['train']]
        cls.pipe=Pipeline([('prepare',preprocessing()),('model',LogisticRegression(max_iter=2000))])
        cls.pipe.fit(cls.train[FEATURES],cls.train[TARGET])
        cls.bundle={'pipeline':cls.pipe,'threshold':.35}

    def test_source_shape_and_dedup(self):
        self.assertEqual(self.audit['raw_rows'],3150)
        self.assertEqual(self.audit['clean_rows'],2850)
        self.assertEqual(self.audit['exact_duplicates_removed'],300)

    def test_no_group_overlap(self):
        groups=[set(self.data.iloc[ix].profile_group) for ix in self.splits.values()]
        for i in range(3):
            for j in range(i):self.assertFalse(groups[i]&groups[j])

    def test_split_partition(self):
        indices=np.concatenate(list(self.splits.values()))
        self.assertEqual(sorted(indices.tolist()),list(range(len(self.data))))

    def test_split_reproducible(self):
        other=split_data(self.data)
        for k in other:np.testing.assert_array_equal(other[k],self.splits[k])

    def test_both_classes_in_every_split(self):
        for idx in self.splits.values():self.assertEqual(set(self.data.iloc[idx][TARGET]),{0,1})

    def test_scaler_fitted_only_train(self):
        scaler=self.pipe.named_steps['prepare'].named_transformers_['numeric'].named_steps['scale']
        np.testing.assert_allclose(scaler.mean_,self.train[NUMERIC].mean())
        self.assertEqual(scaler.n_samples_seen_,len(self.train))

    def test_target_and_status_excluded(self):
        self.assertFalse(set(FEATURES)&{'Churn','Status','Customer Value','Age','Age Group'})

    def test_missing_column_rejected(self):
        with self.assertRaisesRegex(ValueError,'Нет обязательных'):
            validate_features(self.train[FEATURES].drop(columns=FEATURES[0]))

    def test_negative_value_rejected(self):
        x=self.train[FEATURES].head(1).copy(); x.iloc[0,0]=-1
        with self.assertRaisesRegex(ValueError,'Отрицательное'):validate_features(x)

    def test_infinite_value_rejected(self):
        x=self.train[FEATURES].head(1).astype(float); x.iloc[0,0]=np.inf
        with self.assertRaisesRegex(ValueError,'Бесконечное'):validate_features(x)

    def test_bad_numeric_rejected(self):
        x=self.train[FEATURES].head(1).astype(object); x.iloc[0,0]='bad'
        with self.assertRaises(ValueError):validate_features(x)

    def test_missing_and_unseen_category_supported(self):
        x=self.train[FEATURES].head(2).astype(float).copy()
        x.iloc[0,0]=np.nan; x.loc[x.index[1],'Tariff Plan']=99
        p=predict_frame(self.bundle,x).probability
        self.assertTrue(p.between(0,1).all())

    def test_column_order_not_significant(self):
        x=self.train[FEATURES].head(4)
        np.testing.assert_allclose(predict_frame(self.bundle,x),predict_frame(self.bundle,x[FEATURES[::-1]]))

    def test_duplicate_normalized_names_rejected(self):
        with self.assertRaises(ValueError):clean_names(pd.DataFrame([[1,2]],columns=['x y','x  y']))

    def test_threshold_selection_known_case(self):
        threshold,_=select_threshold([0,0,1,1],[.1,.2,.7,.9])
        self.assertAlmostEqual(threshold,.7)

    def test_serialization_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'model.joblib'; joblib.dump(self.bundle,path)
            x=self.train[FEATURES].head(6)
            np.testing.assert_allclose(predict_frame(self.bundle,x),predict_frame(joblib.load(path),x))

    def test_csv_parquet_batches_match(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td); joblib.dump(self.bundle,p/'model.joblib')
            x=self.train[FEATURES].head(17)
            x.to_csv(p/'input.csv',index=False); x.to_parquet(p/'input.parquet',index=False)
            for ext,size in [('csv',4),('parquet',7)]:
                self.assertEqual(predict_file(p/f'input.{ext}',p/f'out_{ext}.csv',p/'model.joblib',size),17)
            a=pd.read_csv(p/'out_csv.csv'); b=pd.read_csv(p/'out_parquet.csv')
            np.testing.assert_allclose(a,b)
            np.testing.assert_allclose(a.probability,predict_frame(self.bundle,x).probability)
            self.assertEqual(a.row_id.tolist(),list(range(17)))

    def test_refuse_overwrite_and_invalid_batch(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'exists.csv'; p.write_text('keep')
            with self.assertRaises(FileExistsError):predict_file('absent.csv',p,'none',3)
            with self.assertRaises(ValueError):predict_file(p,p,'none',3)
            with self.assertRaises(ValueError):predict_file(p,Path(td)/'out.csv','none',0)
            self.assertEqual(p.read_text(),'keep')

if __name__=='__main__':unittest.main(verbosity=2)
