#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 3
parser_transitions.py: Algorithms for completing partial parsess.
Sahil Chopra <schopra8@stanford.edu>
"""

import sys

class PartialParse(object):
    def __init__(self, sentence):
        """Initializes this partial parse.

        @param sentence (list of str): The sentence to be parsed as a list of words.这里的句子是单词的列表
                                        Your code should not modify the sentence.
        """
        # The sentence being parsed is kept for bookkeeping purposes. Do not alter it in your code.
        self.sentence = sentence

        ### YOUR CODE HERE (3 Lines)
        ### Your code should initialize the following fields:初始化数据成员
        ###     self.stack: The current stack represented as a list with the top of the stack as the
        ###                 last element of the list.堆顶对应着list最后
        ###     self.buffer: The current buffer represented as a list with the first item on the
        ###                  buffer as the first item of the list. BUFFER的顶就是句子的第一个词
        ###     self.dependencies: The list of dependencies produced so far. Represented as a list of
        ###             tuples where each tuple is of the form (head, dependent).
        ###             Order for this list doesn't matter.依靠关系是tuple列表
        ###
        ### Note: The root token should be represented with the string "ROOT"
        ###
        self.stack=["ROOT"]
        self.buffer=self.sentence.copy()
        self.dependencies=[]
        ### END YOUR CODE


    def parse_step(self, transition):
        """Performs a single parse step by applying the given transition to this partial parse

        @param transition (str): A string that equals "S", "LA", or "RA" representing the shift,
                                left-arc, and right-arc transitions. You can assume the provided
                                transition is a legal transition.假设transition一定合法，不用检查
        """
        ### YOUR CODE HERE (~7-10 Lines)
        ### TODO:
        ###     Implement a single parsing step, i.e. the logic for the following as
        ###     described in the pdf handout:
        ###         1. Shift
        ###         2. Left Arc
        ###         3. Right Arc
        if transition=="S":
            self.stack.append(self.buffer[0])
            del self.buffer[0]
        elif transition=="LA":#stack的list最后第一个是中心词，把倒数第二个清除
            self.dependencies.append((self.stack[-1],self.stack[-2]))
            del self.stack[-2]
        elif transition=="RA":
            self.dependencies.append((self.stack[-2],self.stack[-1]))
            del self.stack[-1]            
        ### END YOUR CODE

    def parse(self, transitions):
        """Applies the provided transitions to this PartialParse

        @param transitions (list of str): The list of transitions in the order they should be applied

        @return dependencies (list of string tuples): The list of dependencies produced when
                                                        parsing the sentence. Represented as a list of
                                                        tuples where each tuple is of the form (head, dependent).
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    @param sentences (list of list of str): A list of sentences to be parsed
                                            (each sentence is a list of words and each word is of type string)
                                            [['s1w1'，'s1w2'],['s1w1'，'s1w2']]
                                            
    @param model (ParserModel): The model that makes parsing decisions. It is assumed to have a function
                                model.predict(partial_parses) that takes in a list of PartialParses as input and
                                returns a list of transitions predicted for each parse. That is, after calling
                                    transitions = model.predict(partial_parses)
                                transitions[i] will be the next transition to apply to partial_parses[i]. next下一个步骤，不是所有的步骤
                                model有个predict函数，输入是partial_parses的list，返回每个list中每个transition
                                
    @param batch_size (int): The number of PartialParses to include in each minibatch


    @return dependencies (list of dependency lists): A list where each element is the dependencies
                                                    list for a parsed sentence. Ordering should be the
                                                    same as in sentences (i.e., dependencies[i] should
                                                    contain the parse for sentences[i]).
                                                    返回的是一个句子所有的dependencies
    """
    dependencies = []

    ### YOUR CODE HERE (~8-10 Lines)
    ### TODO:
    ###     Implement the minibatch parse algorithm as described in the pdf handout
    ###
    ###     Note: A shallow copy (as denoted in the PDF) can be made with the "=" sign in python, e.g.
    ###                 unfinished_parses = partial_parses[:].
    ###             Here `unfinished_parses` is a shallow copy of `partial_parses`.
    ###             In Python, a shallow copied list like `unfinished_parses` does not contain new instances
    ###             of the object stored in `partial_parses`. Rather both lists refer to the same objects.
    ###             In our case, `partial_parses` contains a list of partial parses. `unfinished_parses`
    ###             contains references to the same objects. Thus, you should NOT use the `del` operator
    ###             to remove objects from the `unfinished_parses` list. This will free the underlying memory that
    ###             is being accessed by `partial_parses` and may cause your code to crash.
    ###             提醒浅拷贝 直接=，但是从unfinished_parses中剔除不可以用del
    '''partial_parses=[PartialParse(sentence) for sentence in sentences]#初始化，将所有的句子全装在partial_parses里了
    unfinished_parses = partial_parses[:]
    while len(unfinished_parses)!=0:
        pred_trans=model.predict(unfinished_parses[:batch_size])#利用model的predict函数输出minibatch所有句子的transitions
        for transition,parser in zip(pred_trans,unfinished_parses[:batch_size]):#zip用在循环中挺常见的
            dependencies.append(parser.parse(transition))#parse确实是将每句话按照dependencies的顺序一点点parse出来，但是transition不是每句话的dependencies序列，而是一步步的
            if len(parser.buffer)==0 and len(parser.stack)==1:
                unfinished_parses.remove(parser)'''#
    #print(sentences)
    partial_parses = [PartialParse(sentence) for sentence in sentences]
    unfinished_parses = partial_parses[:]
    while len(unfinished_parses):
        pred_trans = model.predict(unfinished_parses[:batch_size])#当前两个句子下一步预测出的action
        #print(pred_trans)
        for pred_tran, parse in zip(pred_trans, unfinished_parses[:batch_size]):
            parse.parse_step(pred_tran)#单步进行
            if not parse.buffer and len(parse.stack) == 1:
                unfinished_parses.remove(parse)   # remove的方法，先弄完的先走
                #print('换句子了')
    #由于每个batch内的几句话不是同时结束的，所以只能通过下面的方法一个个的读出dependencies
    dependencies = [partial_parse.dependencies for partial_parse in partial_parses] 
            
        
    ### END YOUR CODE

    return dependencies


def test_step(name, transition, stack, buf, deps,
              ex_stack, ex_buf, ex_deps):
    """Tests that a single parse step returns the expected output"""
    pp = PartialParse([])
    pp.stack, pp.buffer, pp.dependencies = stack, buf, deps

    pp.parse_step(transition)
    stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer), tuple(sorted(pp.dependencies)))
    assert stack == ex_stack, \
        "{:} test resulted in stack {:}, expected {:}".format(name, stack, ex_stack)
    assert buf == ex_buf, \
        "{:} test resulted in buffer {:}, expected {:}".format(name, buf, ex_buf)
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)
    print("{:} test passed!".format(name))


def test_parse_step():
    """Simple tests for the PartialParse.parse_step function
    Warning: these are not exhaustive
    """
    #通过预设一个场景，以及预测一个transition，来检测是否执行顺利
    test_step("SHIFT", "S", ["ROOT", "the"], ["cat", "sat"], [],
              ("ROOT", "the", "cat"), ("sat",), ())
    test_step("LEFT-ARC", "LA", ["ROOT", "the", "cat"], ["sat"], [],
              ("ROOT", "cat",), ("sat",), (("cat", "the"),))
    test_step("RIGHT-ARC", "RA", ["ROOT", "run", "fast"], [], [],
              ("ROOT", "run",), (), (("run", "fast"),))


def test_parse():
    """Simple tests for the PartialParse.parse function
    Warning: these are not exhaustive
    """
    sentence = ["parse", "this", "sentence"]
    dependencies = PartialParse(sentence).parse(["S", "S", "S", "LA", "RA", "RA"])
    dependencies = tuple(sorted(dependencies))
    expected = (('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this'))
    assert dependencies == expected,  \
        "parse test resulted in dependencies {:}, expected {:}".format(dependencies, expected)
    assert tuple(sentence) == ("parse", "this", "sentence"), \
        "parse test failed: the input sentence should not be modified"
    print("parse test passed!")


class DummyModel(object):
    """Dummy model for testing the minibatch_parse function
    First shifts everything onto the stack and then does exclusively right arcs if the first word of
    the sentence is "right", "left" if otherwise.
    """
    #所有的model必须有个predict函数，这里随便设置了一个预测机制
    def predict(self, partial_parses):
        return [("RA" if pp.stack[1] is "right" else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]#这句语法真复杂啊，简便if的嵌套：对这些parses进行循环，分别判断每个的下一步


def test_dependencies(name, deps, ex_deps):
    """Tests the provided dependencies match the expected dependencies"""
    deps = tuple(sorted(deps))
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)


def test_minibatch_parse():
    """Simple tests for the minibatch_parse function
    Warning: these are not exhaustive
    """
    #简单找了四句话，batch_size设置为2，
    sentences = [["right", "arcs", "only"],
                 ["right", "arcs", "only", "again"],
                 ["left", "arcs", "only"],
                 ["left", "arcs", "only", "again"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[1],
                      (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[2],
                      (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
    test_dependencies("minibatch_parse", deps[3],
                      (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))
    print("minibatch_parse test passed!")


if __name__ == '__main__':
    
    args = sys.argv
    #其实这个就是获取用户使用cmd时输入的参数，比如：python test.py part_c，
    #这是args就是有两个元素，第一个是文件自己的地址，第二个是'part_c'
    if len(args) != 2:
        raise Exception("You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
    elif args[1] == "part_c":
        test_parse_step()
        test_parse()
    elif args[1] == "part_d":
        test_minibatch_parse()
    else:
        raise Exception("You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
