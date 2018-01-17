import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
    parser.add_argument('--data', default='data/sick/',
                        help='path to dataset')
    parser.add_argument('--glove', default='data/glove/',
                        help='directory with GLOVE embeddings')
    parser.add_argument('--save', default='checkpoints/',
                        help='directory to save checkpoints in')
    parser.add_argument('--expname', type=str, default='original_model_test',
                        help='Name to identify experiment')
    parser.add_argument('--batchsize', default=50, type=int,
                        help='batchsize for optimizer updates')
    parser.add_argument('--epochs', default=32, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.0045, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=0.007, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings,\
                              incompatible with weight decay')
    parser.add_argument('--optim', default='adagrad',
                        help='optimizer (default: adagrad)')
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    parser.add_argument('--input_dim', default=300,
                        help='input vocab dim (default: 300)')
    parser.add_argument('--mem_dim', default=350,
                        help='mem dim (default: 450)')
    parser.add_argument('--hidden_dim1', default=1000,
                        help='hidden dim 1 (default: 500)')
    parser.add_argument('--hidden_dim2', default=280,
                        help='hidden dim 2 (default: 200)')
    parser.add_argument('--hidden_dim3', default=280,
                        help='hidden dim 3 (default: 120)')
    parser.add_argument('--num_classes', default=5,
                        help='num classes (default: 5)')
    parser.add_argument('--att_hops', default=7,
                        help='attention hops (default: 15)')
    parser.add_argument('--att_units', default=40,
                        help='attention units (default: 150)')
    parser.add_argument('--maxlen', default=40,
                        help='maximum sentence length (default: 40)')
    parser.add_argument('--dropout1', default=0.005,
                        help='dropout1 keep probability (default: 0.15)')
    parser.add_argument('--dropout2', default=0.175,
                        help='dropout2 keep probability (default: 0.15)')
    parser.add_argument('--dropout3', default=0.07,
                        help='dropout3 keep probability (default: 0.15)')
    parser.add_argument('--load_model_params', default=None,
                        help='path to pt.pth model checkpoint')
    parser.add_argument('--mode', default='test',
                        help='Mode in {train, test, eval}')

    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args