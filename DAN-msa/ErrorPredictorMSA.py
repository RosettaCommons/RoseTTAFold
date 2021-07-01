import sys
import argparse
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
import multiprocessing

def main():
    #####################
    # Parsing arguments
    #####################
    parser = argparse.ArgumentParser(description="Error predictor network with predicted distogram",
                                     epilog="v0.0.1")
    parser.add_argument("distogram",
                        action="store",
                        help="predicted distogram (npz format, key for distogram should be 'dist')")
    parser.add_argument("infolder",
                        action="store",
                        help="input folder name full of decoy pdbs having same sequence or path to a single pdb")
    parser.add_argument("outfolder",
                        action="store", nargs=argparse.REMAINDER,
                        help="output folder name. If a pdb path is passed this needs to be a .npz file. Can also be empty. Default is current folder or pdbname.npz")
    parser.add_argument('-prefix', default=None, help='prefix for input pdbs')
    parser.add_argument('--roll', action="store_true", default=False, help="Roll predicted distogram or not [False], should be added if you use distogram from Trunk")
    parser.add_argument("--pdb",
                        "-pdb",
                        action="store_true",
                        default=False,
                        help="Running on a single pdb file instead of a folder (Default: False)")
    parser.add_argument("--leavetemp",
                        "-lt",
                        action="store_true",
                        default=False,
                        help="leaving temporary files (Default: False)")
    parser.add_argument("--verbose",
                        "-v",
                        action="store_true",
                        default=False,
                        help="verbose flag (Default: False)")
    parser.add_argument("--process",
                        "-p", action="store",
                        type=int,
                        default=1,
                        help="# of cpus to use for featurization (Default: 1)")
    parser.add_argument("--gpu",
                        "-g", action="store",
                        type=int,
                        default=0,
                        help="gpu device to use (default gpu0)")
    parser.add_argument("--featurize",
                        "-f",
                        action="store_true",
                        default=False,
                        help="running only featurization (Default: False)")
    parser.add_argument("--reprocess",
                        "-r", action="store_true",
                        default=False,
                        help="reprocessing all feature files (Default: False)")
    args = parser.parse_args()

    ################################
    # Checking file availabilities #
    ################################
    #made outfolder an optional positinal argument. So check manually it's lenght and unpack the string
    if len(args.outfolder)>1:
        print(f"Only one output folder can be specified, but got {args.outfolder}", file=sys.stderr)
        return -1

    if len(args.outfolder)==0:
        args.outfolder = ""
    else:
        args.outfolder = args.outfolder[0]


    if args.infolder.endswith('.pdb'):
        args.pdb = True
    
    if not args.pdb:
        if not isdir(args.infolder):
            print("Input folder does not exist.", file=sys.stderr)
            return -1
        
        #default is current folder
        if args.outfolder == "":
            args.outfolder='.'
        if not isdir(args.outfolder):
            print("Creating output folder:", args.outfolder)
            os.mkdir(args.outfolder)
    else:
        if not isfile(args.infolder):
            print("Input file does not exist.", file=sys.stderr)
            return -1
        
        #default is output name with extension changed to npz
        if args.outfolder == "":
            args.outfolder = os.path.splitext(args.infolder)[0]+".npz"

        if not(".pdb" in args.infolder and ".npz" in args.outfolder):
            print("Input needs to be in .pdb format, and output needs to be in .npz format.", file=sys.stderr)
            return -1
            
    
    script_dir = os.path.dirname(__file__)
    base = os.path.join(script_dir, "models/")
    modelpath = base + "smTr"
        
    if not isdir(modelpath+"_rep1"):
        print("Model checkpoint does not exist", file=sys.stderr)
        return -1
        
    ##############################
    # Importing larger libraries #
    ##############################
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    script_dir = os.path.dirname(__file__)
    sys.path.insert(0, script_dir)
    import pyErrorPred
        
    num_process = 1
    if args.process > 1:
        num_process = args.process
        
    #########################
    # Getting samples names #
    #########################
    if not args.pdb:
        if args.prefix == None:
            samples = [i[:-4] for i in os.listdir(args.infolder) if isfile(args.infolder+"/"+i) and i[-4:] == ".pdb" and i[0]!="."]
            ignored = [i[:-4] for i in os.listdir(args.infolder) if not(isfile(args.infolder+"/"+i) and i[-4:] == ".pdb" and i[0]!=".")]
        else:
            samples = [i[:-4] for i in os.listdir(args.infolder) if isfile(args.infolder+"/"+i) and i[-4:] == ".pdb" and i[0]!="." and args.prefix in i]
            ignored = [i[:-4] for i in os.listdir(args.infolder) if not(isfile(args.infolder+"/"+i) and i[-4:] == ".pdb" and i[0]!=".")]
        if args.verbose: 
            print("# samples:", len(samples))
            if len(ignored) > 0:
                print("# files ignored:", len(ignored))

        ##############################
        # Featurization happens here #
        ##############################
        inputs = [join(args.infolder, s)+".pdb" for s in samples]
        tmpoutputs = [join(args.outfolder, s)+".features.npz" for s in samples]
        if not args.reprocess:
            arguments = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if not isfile(tmpoutputs[i])]
            already_processed = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if isfile(tmpoutputs[i])]
            if args.verbose: 
                print("Featurizing", len(arguments), "samples.", len(already_processed), "are already processed.")
        else:
            arguments = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs))]
            already_processed = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if isfile(tmpoutputs[i])]
            if args.verbose: 
                print("Featurizing", len(arguments), "samples.", len(already_processed), "are re-processed.")

        if num_process == 1:
            for a in arguments:
                pyErrorPred.process(a)
        else:
            pool = multiprocessing.Pool(num_process)
            out = pool.map(pyErrorPred.process, arguments)

        print(modelpath)
        # Exit if only featurization is needed
        if args.featurize:
            return 0
        
        ###########################
        # Prediction happens here #
        ###########################
        samples = [s for s in samples if isfile(join(args.outfolder, s+".features.npz"))]
        disto = np.load(args.distogram)['dist'].astype(np.float32)
        if args.roll:
            disto  = np.roll(disto,1,axis=-1)

        pyErrorPred.predict(samples,
                            disto,
                            modelpath,
                            args.outfolder,
                            verbose=args.verbose)

        if not args.leavetemp:
            pyErrorPred.clean(samples,
                              args.outfolder,
                              verbose=args.verbose)
            
    # Processing for single sample
    else:
        infilepath = args.infolder
        outfilepath = args.outfolder
        infolder = "/".join(infilepath.split("/")[:-1])
        insamplename = infilepath.split("/")[-1][:-4]
        outfolder = "/".join(outfilepath.split("/")[:-1])
        outsamplename = outfilepath.split("/")[-1][:-4]
        feature_file_name = join(outfolder, outsamplename+".features.npz")
        if args.verbose: 
            print("only working on a file:", outfolder, outsamplename)
        # Process if file does not exists or reprocess flag is set
        
        if (not isfile(feature_file_name)) or args.reprocess:
            pyErrorPred.process((join(infolder, insamplename+".pdb"),
                                feature_file_name,
                                args.verbose))   
        if isfile(feature_file_name):
            disto = np.load(args.distogram)['dist'].astype(np.float32)
            if args.roll:
                disto  = np.roll(disto,1,axis=-1)
            pyErrorPred.predict([outsamplename],
                    disto,
                    modelpath,
                    outfolder,
                    verbose=args.verbose)
            
            if not args.leavetemp:
                pyErrorPred.clean([outsamplename],
                                  outfolder,
                                  verbose=args.verbose)
        else:
            print(f"Feature file does not exist: {feature_file_name}", file=sys.stderr)
if __name__== "__main__":
    main()
