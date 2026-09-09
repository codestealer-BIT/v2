import os
import sys
sys.path.append('/mnt/bn/jiny-ttls-i18n-fr1q/ptx__')
import matx
import json
import random
from typing import List, Callable
from text_cutter.cut import Cutter
from ptx.matx.vocabulary import VocabularyWithOOVHash
from ptx.matx.pipeline import Pipeline, BertInputsBuilder, Text2Tokens
from ptx.matx.tokenizer import WordPieceTokenizer, SentencePieceTokenizer


CONV_OP_ALIAS = {
    'char2pinyin': 'CharPinyinOp',
    'demoji': 'DemojiOp',
    'filteremoji': 'FilterEmojiOp',
    'lower_str': 'LowerStrOp',
    'remove_punc': 'RemovePuncOp',
    'replace_str': 'ReplaceStrOp',
    'text2pinyin': 'SpamInputOp',
    'text2pinyinsecond': 'SpamInputV2Op',
    'sub_str': 'SubStrOp',
    'zht2zhs': 'Zht2ZhsOp',
}

class SimpleCutter:
    def __init__(self, cut_type: str, location: str, cut_level: str) -> None:
        self.cutter: Callable = Cutter(cut_type, location)
        self.cut_level: str = cut_level

    def __call__(self, text: str) -> str:
        words: List[str] = self.cutter(text, self.cut_level)
        return ' '.join(words)


class Tokenizer_Pipeline():
    def __init__(self, tokenizer_dir, max_text_len, do_mask=False):
        self.max_text_len = max_text_len
        options = json.load(open(tokenizer_dir+'/config.json'))
        option = options['pipeline']

        resource_path = option.get('resource_path', tokenizer_dir)
        cut_option = option.get('cutter')
        tokenize_option = option['tokenizer']
        vocab_file = os.path.join(resource_path, option['vocab'])
        vocab_oov_buckets = option.get('vocab_oov_buckets', 0)
        special_tokens = option.get('special_tokens', {})

        if do_mask:
            if 'mask' not in special_tokens:
                special_tokens['mask'] = '<mask>'

        max_len = option['max_len']

        self.conv_enabled = bool(option.get('conv'))
        # self.ConvOps = [self._create_conv_op(co) for co in option.get('conv', [])]
        ConvOps = [self._create_conv_op(co) for co in option.get('conv', [])]
        self.conv_before_cut = option.get('conv_before_cut', False)

        self.preprocess_use_parallel = option.get('preprocess_use_parallel', False)
        self.dynamic_max_len = option.get('dynamic_max_len', False)

        Cutter = matx.script(SimpleCutter)(
            cut_option['cut_type'],
            cut_option['location'],
            cut_option['cut_level'],
        )

        if tokenize_option['type'] == 'wp':
            Tokenizer = WordPieceTokenizer(
                os.path.join(resource_path, tokenize_option['vocab']),
                lower_case=tokenize_option['lower_case'],
            ).tokenizer
        elif tokenize_option['type'] == 'sp':
            Tokenizer = SentencePieceTokenizer(
                os.path.join(resource_path, tokenize_option['vocab']),
                resource_type=tokenize_option.get('resource_type', 'model'),
                norm_form=tokenize_option.get('norm_form', 'nfkc'),
            ).tokenizer
        else:
            raise Exception(f'Unsupported matx tokenizer op: {tokenize_option["type"]}')

        self.vocab = matx.script(VocabularyWithOOVHash)(vocab_file, num_oov_buckets=vocab_oov_buckets)
        self.special_tokens = special_tokens
        self.max_len = min(max_len, self.max_text_len) # min

        self.text2tokens = matx.script(Text2Tokens)(
            Cutter, Tokenizer,
            ConvOps, self.conv_enabled, self.conv_before_cut,
            self.preprocess_use_parallel,
        )

        self.InputBuilder = matx.script(BertInputsBuilder)(
            self.vocab, self.special_tokens['pad'], self.special_tokens['unk'],
            self.special_tokens['cls'], self.special_tokens['sep'], self.max_len,
            # self.dynamic_max_len,
        )
        self.PAD = self.vocab(self.special_tokens['pad'])
        self.SEP = self.vocab(self.special_tokens['sep'])
        # print(self.PAD, self.SEP, self.vocab(self.special_tokens['cls']))
        # print('pad is {}'.format(self.PAD))

    def _create_conv_op(self, op_config):
        import rtc_ops

        if isinstance(op_config, str):
            return matx.script(getattr(rtc_ops, self._get_conv_op_name(op_config)))()
        if isinstance(op_config, dict):
            op_name = self._get_conv_op_name(op_config['name'])
            op_params = op_config.get('params', {})
            return matx.script(getattr(rtc_ops, op_name))(**op_params)
        raise Exception(f'Invalid conv op config: {op_config}')

    def _get_conv_op_name(self, op_name: str) -> str:
        return CONV_OP_ALIAS.get(op_name, op_name)

    def tokenize(self, text: List[str]):
        tokens = self.text2tokens(text)
        token_ids = self.InputBuilder(tokens).asnumpy()[0]
        return token_ids

    def _get_word_indexes_by_tokens(self, tokens: List[str]) -> List[int]:
        word_indexes = []
        i = -1
        for t in tokens:
            if t.startswith('▁') or t in list(self.special_tokens.values()):
                i += 1
            word_indexes.append(i)
        return word_indexes

    def do_mask(self, input_tokens, max_predictions_per_seq=16, masked_lm_prob=0.16, rnd=random.Random()):

        output_tokens = list(input_tokens)
        output_labels = [-1] * len(input_tokens)
        all_masked_tokens = []
        candidate_info = []
        spec_num = 0
        cur_start, cur_length = 0, 0

        word_infos = self._get_word_indexes_by_tokens(input_tokens)
        for i, token in enumerate(input_tokens):
            if token in list(self.special_tokens.values()):
                spec_num += 1
                continue
            elif word_infos[i] != word_infos[i - 1]:
                if cur_length > 0:
                    candidate_info.append([cur_start, cur_length])
                cur_start = i
                cur_length = 1
            else:
                cur_length += 1
        if cur_length > 0:
            candidate_info.append([cur_start, cur_length])
        rnd.shuffle(candidate_info)
        num_to_predict = min(max_predictions_per_seq, max(1, int(round((len(input_tokens) - spec_num) * 0.25))))

        slots = 0
        has_mask = False
        for [start, length] in candidate_info:
            if slots >= num_to_predict:
                break
            if slots + length > num_to_predict:
                continue
            prob = rnd.random()
            gate = masked_lm_prob
            if prob > gate:
                continue
            if length == 2 and input_tokens[start] == '▁' and input_tokens[start + 1] in ['@', '#']:
                if prob > 0.075:
                    continue
            has_mask = True

            all_masked_tokens.extend([[ti, input_tokens[ti]] for ti in range(start, start + length)])
            masked_tokens = [self.special_tokens['mask']] * length

            slots += length
            for i in range(start, start + length):
                output_labels[i] = self.vocab(output_tokens[i])
                output_tokens[i] = masked_tokens[i - start]

        if not has_mask and len(input_tokens) - spec_num > 3:
            for _ in range(10):
                mask_idx = rnd.choice(list(range(len(input_tokens))))
                if input_tokens[mask_idx] not in self.special_tokens:
                    output_labels[mask_idx] = self.vocab(output_tokens[mask_idx])
                    all_masked_tokens.extend([mask_idx, input_tokens[mask_idx]])
                    output_tokens[mask_idx] = self.special_tokens['mask']
                    break

        return output_tokens, output_labels, all_masked_tokens


if __name__ == "__main__":
    tokenizer_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/models/byted_nlp_model/m_albertv2_base_v2_t858845_879315'
    tokenizer = Tokenizer_Pipeline(tokenizer_dir, 128)
    print(tokenizer.tokenize(['hello world, I love you china']))