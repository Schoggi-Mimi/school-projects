import gc

class NoiseFilter():
    def _filter_1(self, df):
        return df.query('reads > 100 & signal_to_noise > 0.75')
    
    def _filter_2(self, df):
        return df.query('SN_filter == 1')

    def apply_filters(self, df, verbose=True):
        prev_len = len(df)
        
        #Filter 1 is redundant b.c. SN_filter==1 only if signal_to_noise> 1.00 and reads> 100
        #df = self._filter_1(df)

        df = self._filter_2(df)
        
        gc.collect()

        if verbose:
            new_len = len(df)
            print(f'Removed n-samples: {prev_len - new_len}')
            print(f'Remaining samples: {new_len}')
            
        return df