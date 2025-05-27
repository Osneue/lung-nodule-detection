from rknn.api import RKNN


class RKNN_model_container():
    def __init__(self, model_path, target=None, device_id=None, \
                 perf_debug=False, eval_mem=False) -> None:
        rknn = RKNN()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        if target==None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(target=target, device_id=device_id, \
                                    perf_debug=perf_debug, eval_mem=eval_mem)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')
        
        self.rknn = rknn

    # def __del__(self):
    #     self.release()

    def run(self, inputs, data_format=None):
        if self.rknn is None:
            print("ERROR: rknn has been released")
            return []

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs, data_format=data_format)
    
        return result

    def eval(self):
        rknn = self.rknn
        # eval perf
        print('--> Eval perf')
        rknn.eval_perf()

        # eval perf
        print('--> Eval memory')
        rknn.eval_memory()

    def release(self):
        self.rknn.release()
        self.rknn = None