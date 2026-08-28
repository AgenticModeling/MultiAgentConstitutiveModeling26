import os
import numpy
import random
import pandas
import scipy.io


class Dataloader():

    def __init__(self, config):
        self._config = config
        self._data   = {}


    def load(self):
        if   self._config["problem"] == "synthetic_rubber":
            self._load_synthetic_rubber()
        elif self._config["problem"] == "experimental_rubber":
            self._load_experimental_rubber()
        elif self._config["problem"] == "experimental_brain":
            self._load_experimental_brain()
        elif self._config["problem"] == "experimental_skin":
            self._load_experimental_skin()
        else:
            raise NotImplementedError(f"Problem '{self._config['problem']}' is not implemented.")


    def get_data(self):
        return self._data 


    def translate_brain_loading_code(self, code):
        if   code == 0:
            return "tens"
        elif code == 1:
            return "comp"
        elif code == 2:
            return "shear"
        else:
            raise NotImplementedError(f"Loading code '{code}' is not implemented.")


    def _define_brain_loading_code(self, loading):
        if   loading == "tens":
            return 0
        elif loading == "comp":
            return 1
        elif loading == "shear":
            return 2
        else:
            raise NotImplementedError(f"Loading type '{loading}' is not implemented.")


    def _compute_F(self, loading, lam):
        lam = lam.astype(numpy.float32)
        if   self._config["problem"] == "experimental_skin":
            F = numpy.zeros([len(lam), 3, 3])
            F[:,0,0] = lam[:,0]
            F[:,1,1] = lam[:,1]
            F[:,2,2] = 1.0/(lam[:,0]*lam[:,1])
        elif loading == "uni-x" or \
             loading == "tens"  or \
             loading == "comp":
            F = numpy.zeros([len(lam), 3, 3])
            F[:,0,0] = lam
            F[:,1,1] = 1.0/(numpy.sqrt(lam))
            F[:,2,2] = 1.0/(numpy.sqrt(lam))
        elif loading == "equi-x":
            F = numpy.zeros([len(lam), 3, 3])
            F[:,0,0] = lam
            F[:,1,1] = lam
            F[:,2,2] = 1.0/lam**2
        elif loading == "strip-x":
            F = numpy.zeros([len(lam), 3, 3])
            F[:,0,0] = lam
            F[:,1,1] = 1.0
            F[:,2,2] = 1/lam
        elif loading == "shear":
            F = numpy.zeros([len(lam), 3, 3])
            F[:,0,0] = 1
            F[:,1,1] = 1
            F[:,2,2] = 1
            F[:,0,1] = lam
        else:
            raise NotImplementedError(f"Loading type '{loading}' is not implemented.")

        return F.astype(numpy.float32)


    def _load_synthetic_rubber(self):
        path           = os.path.join("..", "input", "rubber_synthetic", "GenMR_F_P.npy")
        data           = numpy.load(path, allow_pickle=True)[()]
        data["equi-x"] = data.pop("equi")

        self._split_synthetic_rubber(data=data, data_dict=self._data, loading="uni-x",   test_data_indices=[ 2, 3, 5])
        self._split_synthetic_rubber(data=data, data_dict=self._data, loading="equi-x",  test_data_indices=[ 5,10,11])
        self._split_synthetic_rubber(data=data, data_dict=self._data, loading="strip-x", test_data_indices=[12,13,14])

        self._data["all"] = {
            "F": {"all": [], "train": [], "test": []},
            "P": {"all": [], "train": [], "test": []}
        }
        for loading in data.keys():
            for key in self._data["all"]["F"].keys():
                self._data["all"]["F"][key].append(self._data[loading]["F"][key])
                self._data["all"]["P"][key].append(self._data[loading]["P"][key])
        for key in self._data["all"]["F"].keys():
            self._data["all"]["F"][key] = numpy.concatenate(self._data["all"]["F"][key], axis=0)
            self._data["all"]["P"][key] = numpy.concatenate(self._data["all"]["P"][key], axis=0)


    def _split_synthetic_rubber(self, data, data_dict, loading, test_data_indices):
        def nan_array_like(arr):
            return numpy.full_like(arr, numpy.nan, dtype=numpy.float32)

        train_data_mask = numpy.ones( shape=(data[loading][0].shape[0],), dtype=bool)
        test_data_mask  = numpy.zeros(shape=(data[loading][0].shape[0],), dtype=bool)
        train_data_mask[test_data_indices] = False
        test_data_mask[ test_data_indices] = True

        all_F   = data[loading][0][:,              :,:]
        train_F = data[loading][0][train_data_mask,:,:]
        test_F  = data[loading][0][test_data_mask, :,:]
        all_P   = data[loading][1][:,              0,0]
        train_P = data[loading][1][train_data_mask,0,0]
        test_P  = data[loading][1][test_data_mask, 0,0]
        all_loading_code   = nan_array_like(  all_P)
        train_loading_code = nan_array_like(train_P)
        test_loading_code  = nan_array_like( test_P)
        
        data_dict[loading]      = {}
        data_dict[loading]["F"] = {}
        data_dict[loading]["P"] = {}
        data_dict[loading]["F"]["all"]   =   all_F.astype(numpy.float32)
        data_dict[loading]["F"]["train"] = train_F.astype(numpy.float32)
        data_dict[loading]["F"]["test"]  =  test_F.astype(numpy.float32)
        data_dict[loading]["P"]["all"]   = numpy.stack([  all_P,   all_loading_code], axis=1).astype(numpy.float32)
        data_dict[loading]["P"]["train"] = numpy.stack([train_P, train_loading_code], axis=1).astype(numpy.float32)
        data_dict[loading]["P"]["test"]  = numpy.stack([ test_P,  test_loading_code], axis=1).astype(numpy.float32)


    def _load_experimental_rubber(self):
        path = os.path.join("..", "input", "rubber_experimental", "Treloar_result.mat")
        data = scipy.io.loadmat(path)

        self._split_experimental_rubber(data=data, data_dict=self._data, data_loading="train_ut", data_dict_loading="uni-x",   test_data_indices=[ 1, 2,10   ])
        self._split_experimental_rubber(data=data, data_dict=self._data, data_loading="train_bt", data_dict_loading="equi-x",  test_data_indices=[ 2,10      ])
        self._split_experimental_rubber(data=data, data_dict=self._data, data_loading="train_ps", data_dict_loading="strip-x", test_data_indices=[ 5, 7, 8, 9])

        self._data["all"] = {
            "F": {"all": [], "train": [], "test": []},
            "P": {"all": [], "train": [], "test": []}
        }
        for loading in ["uni-x", "equi-x", "strip-x"]:
            for key in self._data["all"]["F"].keys():
                self._data["all"]["F"][key].append(self._data[loading]["F"][key])
                self._data["all"]["P"][key].append(self._data[loading]["P"][key])
        for key in self._data["all"]["F"].keys():
            self._data["all"]["F"][key] = numpy.concatenate(self._data["all"]["F"][key], axis=0)
            self._data["all"]["P"][key] = numpy.concatenate(self._data["all"]["P"][key], axis=0)


    def _split_experimental_rubber(self, data, data_dict, data_loading, data_dict_loading, test_data_indices):
        def nan_array_like(arr):
            return numpy.full_like(arr, numpy.nan, dtype=numpy.float32)

        train_data_mask = numpy.ones( shape=(data[f"{data_loading}_lam"].shape[1],), dtype=bool)
        test_data_mask  = numpy.zeros(shape=(data[f"{data_loading}_lam"].shape[1],), dtype=bool)
        train_data_mask[test_data_indices] = False
        test_data_mask[ test_data_indices] = True

        all_F   = self._compute_F(data_dict_loading, data[f"{data_loading}_lam"].reshape((-1,))[:              ])
        train_F = self._compute_F(data_dict_loading, data[f"{data_loading}_lam"].reshape((-1,))[train_data_mask])
        test_F  = self._compute_F(data_dict_loading, data[f"{data_loading}_lam"].reshape((-1,))[test_data_mask ])
        all_P   = data[f"{data_loading}_P11"].reshape((-1,))[:              ]
        train_P = data[f"{data_loading}_P11"].reshape((-1,))[train_data_mask]
        test_P  = data[f"{data_loading}_P11"].reshape((-1,))[test_data_mask ]
        all_loading_code   = nan_array_like(  all_P)
        train_loading_code = nan_array_like(train_P)
        test_loading_code  = nan_array_like( test_P)
        
        data_dict[data_dict_loading]      = {}
        data_dict[data_dict_loading]["F"] = {}
        data_dict[data_dict_loading]["P"] = {}
        data_dict[data_dict_loading]["F"]["all"]   =   all_F.astype(numpy.float32)
        data_dict[data_dict_loading]["F"]["train"] = train_F.astype(numpy.float32)
        data_dict[data_dict_loading]["F"]["test"]  =  test_F.astype(numpy.float32)
        data_dict[data_dict_loading]["P"]["all"]   = numpy.stack([  all_P,   all_loading_code], axis=1).astype(numpy.float32)
        data_dict[data_dict_loading]["P"]["train"] = numpy.stack([train_P, train_loading_code], axis=1).astype(numpy.float32)
        data_dict[data_dict_loading]["P"]["test"]  = numpy.stack([ test_P,  test_loading_code], axis=1).astype(numpy.float32)


    def _load_experimental_brain(self):
        path          = os.path.join("..", "input", "brain_experimental", "Brain_F_P.npy")
        data          = numpy.load(path, allow_pickle=True)[()]
        data["shear"] = data.pop("simple_shear")

        self._split_experimental_brain(data=data, data_dict=self._data, loading="tens",  test_data_indices=[0,1,5])
        self._split_experimental_brain(data=data, data_dict=self._data, loading="comp",  test_data_indices=[0,1,5])
        self._split_experimental_brain(data=data, data_dict=self._data, loading="shear", test_data_indices=[0,1,5])

        self._data["all"] = {
            "F": {"all": [], "train": [], "test": []},
            "P": {"all": [], "train": [], "test": []}
        }
        for loading in data.keys():
            for key in self._data["all"]["F"].keys():
                self._data["all"]["F"][key].append(self._data[loading]["F"][key])
                self._data["all"]["P"][key].append(self._data[loading]["P"][key])
        for key in self._data["all"]["F"].keys():
            self._data["all"]["F"][key] = numpy.concatenate(self._data["all"]["F"][key], axis=0)
            self._data["all"]["P"][key] = numpy.concatenate(self._data["all"]["P"][key], axis=0)


    def _split_experimental_brain(self, data, data_dict, loading, test_data_indices):
        def brain_loading_code_like(arr):
            return numpy.full_like(arr, self._define_brain_loading_code(loading), dtype=numpy.float32)

        train_data_mask = numpy.ones( shape=(data[loading][0].shape[0],), dtype=bool)
        test_data_mask  = numpy.zeros(shape=(data[loading][0].shape[0],), dtype=bool)
        train_data_mask[test_data_indices] = False
        test_data_mask[ test_data_indices] = True

        all_F   = data[loading][0].numpy()[:,              :,:]
        train_F = data[loading][0].numpy()[train_data_mask,:,:]
        test_F  = data[loading][0].numpy()[test_data_mask, :,:]
        all_P   = data[loading][1].numpy()[:,              0]
        train_P = data[loading][1].numpy()[train_data_mask,0]
        test_P  = data[loading][1].numpy()[test_data_mask, 0]
        all_loading_code   = brain_loading_code_like(  all_P)
        train_loading_code = brain_loading_code_like(train_P)
        test_loading_code  = brain_loading_code_like( test_P)
        
        data_dict[loading]      = {}
        data_dict[loading]["F"] = {}
        data_dict[loading]["P"] = {}
        data_dict[loading]["F"]["all"]   =   all_F.astype(numpy.float32)
        data_dict[loading]["F"]["train"] = train_F.astype(numpy.float32)
        data_dict[loading]["F"]["test"]  =  test_F.astype(numpy.float32)
        data_dict[loading]["P"]["all"]   = numpy.stack([  all_P,   all_loading_code], axis=1).astype(numpy.float32)
        data_dict[loading]["P"]["train"] = numpy.stack([train_P, train_loading_code], axis=1).astype(numpy.float32)
        data_dict[loading]["P"]["test"]  = numpy.stack([ test_P,  test_loading_code], axis=1).astype(numpy.float32)


    def _load_experimental_skin(self):
        path             = os.path.join("..", "input", "skin_experimental", "NODE_porcine_skin_data_1.csv")
        all_data         = pandas.read_csv(path, delimiter=",", header="infer", index_col=0)
        all_data.columns = ["lam_x", "lam_y", "sigma_x", "sigma_y"]

        indices = {}
        indices["strip-x"] = [229, 330]
        indices["off-x"]   = [ 72, 148]
        indices["biaxial"] = [148, 229]
        indices["off-y"  ] = [  0,  72]
        indices["strip-y"] = [330, 402]

        data = {}
        for loading in indices.keys():
            start         = indices[loading][0]
            end           = indices[loading][1]
            data[loading] = all_data.iloc[start:end,:]

        random.seed(42)
        for loading in data.keys():
            data_n            = len(data[loading])
            test_data_n       = round(0.2*data_n)
            test_data_indices = random.sample(range(data_n), test_data_n)
            self._split_experimental_skin(data=data, data_dict=self._data, loading=loading, test_data_indices=test_data_indices)

        self._data["all"] = {
            "F":     {"all": [], "train": [], "test": []},
            "Sigma": {"all": [], "train": [], "test": []}
        }
        for loading in data.keys():
            for key in self._data["all"]["F"].keys():
                self._data["all"]["F"]    [key].append(self._data[loading]["F"]    [key])
                self._data["all"]["Sigma"][key].append(self._data[loading]["Sigma"][key])
        for key in self._data["all"]["F"].keys():
            self._data["all"]["F"]    [key] = numpy.concatenate(self._data["all"]["F"]    [key], axis=0)
            self._data["all"]["Sigma"][key] = numpy.concatenate(self._data["all"]["Sigma"][key], axis=0)


    def _split_experimental_skin(self, data, data_dict, loading, test_data_indices):
        train_data_mask = numpy.ones( shape=(data[loading].shape[0],), dtype=bool)
        test_data_mask  = numpy.zeros(shape=(data[loading].shape[0],), dtype=bool)
        train_data_mask[test_data_indices] = False
        test_data_mask[ test_data_indices] = True

        all_lam   = data[loading][["lam_x", "lam_y"]].to_numpy()[:              ]
        train_lam = data[loading][["lam_x", "lam_y"]].to_numpy()[train_data_mask]
        test_lam  = data[loading][["lam_x", "lam_y"]].to_numpy()[test_data_mask ]
        all_F   = self._compute_F(loading,   all_lam)
        train_F = self._compute_F(loading, train_lam)
        test_F  = self._compute_F(loading,  test_lam)
        all_sigma_x   = data[loading]["sigma_x"].to_numpy()[:              ]
        train_sigma_x = data[loading]["sigma_x"].to_numpy()[train_data_mask]
        test_sigma_x  = data[loading]["sigma_x"].to_numpy()[test_data_mask ]
        all_sigma_y   = data[loading]["sigma_y"].to_numpy()[:              ]
        train_sigma_y = data[loading]["sigma_y"].to_numpy()[train_data_mask]
        test_sigma_y  = data[loading]["sigma_y"].to_numpy()[test_data_mask ]

        data_dict[loading]          = {}
        data_dict[loading]["F"]     = {}
        data_dict[loading]["Sigma"] = {}
        data_dict[loading]["F"]    ["all"]   =   all_F.astype(numpy.float32)
        data_dict[loading]["F"]    ["train"] = train_F.astype(numpy.float32)
        data_dict[loading]["F"]    ["test"]  =  test_F.astype(numpy.float32)
        data_dict[loading]["Sigma"]["all"]   = numpy.stack([  all_sigma_x,   all_sigma_y,   all_lam[:,0],   all_lam[:,1]], axis=1).astype(numpy.float32)
        data_dict[loading]["Sigma"]["train"] = numpy.stack([train_sigma_x, train_sigma_y, train_lam[:,0], train_lam[:,1]], axis=1).astype(numpy.float32)
        data_dict[loading]["Sigma"]["test"]  = numpy.stack([ test_sigma_x,  test_sigma_y,  test_lam[:,0],  test_lam[:,1]], axis=1).astype(numpy.float32)
