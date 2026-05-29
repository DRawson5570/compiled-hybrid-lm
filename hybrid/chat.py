#!/usr/bin/env python3
"""CMI Production Chatbot — interactive terminal chat using V4 DeepCausalLM.

Usage:
    python hybrid/chat.py                    # interactive mode
    python hybrid/chat.py --prompt "Hello"   # single prompt
    python hybrid/chat.py --model artifacts/steerer_v4_sem_learned/steerer_best_b.pt  # custom model
"""
import argparse, sys, time; sys.path.insert(0, '.')
from hybrid.chatbot import ProductionChatbot

def main():
    p = argparse.ArgumentParser(description='CMI Production Chatbot')
    p.add_argument('--model', default='artifacts/steerer_v4/steerer_best_b.pt')
    p.add_argument('--chat-cartridge', default='artifacts/steerer_chat_production_v5_balanced_b384/chat_cartridge.pt')
    p.add_argument('--temperature', type=float, default=0.3)
    p.add_argument('--max-tokens', type=int, default=80)
    p.add_argument('--prompt', action='append')
    args = p.parse_args()

    print('Loading CMI chatbot...', end='', flush=True)
    t0 = time.perf_counter()
    bot = ProductionChatbot(
        base_model=args.model, general_steerer=args.model,
        chat_cartridge=args.chat_cartridge,
    )
    params = sum(p.numel() for p in bot.model.parameters())
    print(f' {params:,} params, {time.perf_counter()-t0:.1f}s')
    print(f'CMI ready. Type /quit to exit.\n')

    if args.prompt:
        for prompt in args.prompt:
            answer, elapsed, toks = bot.generate(
                prompt, temperature=args.temperature, max_new_tokens=args.max_tokens)
            tps = toks / max(elapsed, 0.001)
            print(f'You: {prompt}')
            print(f'CMI: {answer}')
            print(f'     [{toks} tok, {elapsed:.1f}s, {tps:.0f} tok/s]\n')
        bot.cleanup()
        return

    turns = total_tokens = total_time = 0
    while True:
        try:
            user = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user in ('/quit', '/exit'):
            break
        if not user:
            continue
        answer, elapsed, toks = bot.generate(
            user, temperature=args.temperature, max_new_tokens=args.max_tokens)
        tps = toks / max(elapsed, 0.001)
        print(f'CMI: {answer}')
        print(f'     [{toks}t/{elapsed:.1f}s/{tps:.0f}tok/s]')
        turns += 1; total_tokens += toks; total_time += elapsed

    if turns:
        print(f'\nSession: {turns} turns, {total_tokens} tok, {total_time:.1f}s, '
              f'{total_tokens/max(total_time,0.001):.0f} tok/s avg')
    bot.cleanup()

if __name__ == '__main__':
    main()
