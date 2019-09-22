from __future__ import print_function
import torch
from options import get_testing_parser, parse_args
from hncm_dataloader import HNCMDataLoader
from hncm_model import  HNCMModel

def test(args):
	# load data
    data = HNCMDataLoader(args)
    tst_refs = data.tst[-1]
    trg_vocab = data.vocab
    print('before load arg', args)
	# load model
    model = HNCMModel.build_model(args, data.vocab)
    model = model.load(args.load_model_from)
    print('after load arg', model.args)

    if args.cuda:
        model = model.cuda()

    all_hyps, all_scores = [], []
    for sample in HNCMDataLoader.data_iter(data.tst, data.vocab, batch_size=args.batch_size, shuffle=False, cuda=args.cuda):
        # hyps: [batch, beam_size, num_seq, max_decoding_words], scores: [batch, beam_size]
        hyps, scores = model.generate(sample['src_seq'], sample['src_lengths'], sample['fact_seq'],
                                      sample['fact_lengths'], beam_size=args.beam_size,
                                      decode_max_length=args.decode_max_length, to_word=True)
        all_hyps.extend(hyps)
        all_scores.extend(scores)
    tst_hyps = []
    with open(args.save_decode_file, 'w') as f:
        for hid, (hyps, scores) in enumerate(list(zip(all_hyps, all_scores))):
            best_beam = hyps[0]
            sent = " ".join(best_beam[:-1])
            f.write(sent + '\n')
            tst_hyps.append(sent)

if __name__ == '__main__':
    parser = get_testing_parser()
    parser = HNCMDataLoader.add_args(parser)
    parser = HNCMModel.add_args(parser)
    args = parser.parse_args()
    test(args)
