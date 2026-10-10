"""Observe existing autograd edges without adding or replacing graph operations."""
import itertools

import torch

from check import metric


class V0Edges:
    def __init__(self, loss, source):
        self.source = source
        self.source.retain_grad()
        self.records = []
        self.handles = []
        producer = source.grad_fn
        pending = [loss.grad_fn]
        nodes = set()
        while pending:
            node = pending.pop()
            if node is None or node in nodes:
                continue
            nodes.add(node)
            pending.extend(child for child, _ in node.next_functions)
        for node in nodes:
            edges = [i for i, (child, output_nr) in enumerate(node.next_functions)
                     if child is producer and output_nr == source.output_nr]
            name = node.name()
            if not edges and not name.startswith('LerpBackward'):
                continue
            def capture(grad_inputs, grad_outputs, name=name, edges=edges):
                if edges:
                    for i in edges:
                        if grad_inputs[i] is not None:
                            self.records.append({'kind': 'source-edge', 'node': name,
                                                 'gradient': grad_inputs[i].detach().cpu().clone()})
                else:
                    self.records.append({'kind': 'lerp-end', 'node': name,
                                         'gradient': grad_inputs[1].detach().cpu().clone()})
            self.handles.append(node.register_hook(capture))

    def finish(self):
        result = {'total': self.source.grad.detach().cpu().clone(), 'records': self.records}
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.source = None
        return result


def summarize(case):
    parts = [r['gradient'] for r in case['records'] if r['kind'] == 'source-edge']
    exact = sum(p.double() for p in parts)
    sequential = parts[0].clone()
    for part in parts[1:]:
        sequential = sequential + part
    future = next((r['gradient'] for r in case['records']
                   if r['kind'] == 'source-edge' and
                   ('Gather' in r['node'] or 'StackBackward' in r['node'])), None)
    lerps = [r['gradient'] for r in case['records'] if r['kind'] == 'lerp-end']
    result = {'source_edges_in_execution_order': [r['node'] for r in case['records']
                                                 if r['kind'] == 'source-edge'],
              'total_vs_fp64_sum': metric(case['total'], exact),
              'total_matches_execution_order_sum': torch.equal(case['total'], sequential),
              'permutation_matches': [], 'later_layer_branches': len(lerps)}
    # Direct official v_first has one edge per later layer. Factorial search is
    # useful only for the two/three-way historical carrier diagnostic.
    result['permutation_search_performed']=len(parts)<=6
    orders=itertools.permutations(range(len(parts))) if len(parts)<=6 else ()
    for order in orders:
        folded = parts[order[0]].clone()
        for i in order[1:]:
            folded = folded + parts[i]
        if torch.equal(folded, case['total']):
            result['permutation_matches'].append(list(order))
    if future is not None and lerps:
        # All diagnostics here use time-major [T,B,D]. End gradients are either
        # this shape or the corresponding tensor-parallel full slice.
        exact_future = sum(p.double().reshape_as(future) for p in lerps)
        future_fold = lerps[0].reshape_as(future).clone()
        for part in lerps[1:]:
            future_fold = future_fold + part.reshape_as(future)
        result['future_vs_fp64_branch_sum'] = metric(future, exact_future)
        result['future_matches_reverse_layer_bf16_sum'] = torch.equal(future, future_fold)
        local_parts=[r['gradient'] for r in case['records']
                     if r['kind']=='source-edge' and
                     'Gather' not in r['node'] and 'StackBackward' not in r['node']]
        all_branches=exact_future+sum(p.double() for p in local_parts)
        result['total_vs_fp64_all_branch_sum']=metric(case['total'],all_branches)
        result['one_final_bf16_round_vs_fp64_all_branch_sum']=metric(all_branches.to(case['total'].dtype),all_branches)
    return result
