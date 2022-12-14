try:
    from urllib import urlencode
except ImportError:
    from urllib.parse import urlencode


import pickle
import requests
def get_id(idx):
    return int(idx[1:])

class Symbolics():

    def __init__(self, seq, mode='online', url=None):
        if mode != 'online':
            print("loading knowledge base...")
            self.graph = pickle.load(open('wikidata.pkl', 'rb'))
            self.type_dict = pickle.load(open('type_kb.pkl', 'rb'))
            print("Load done!")
        else:
            self.graph = None
            self.type_dict = None

        self.seq = seq
        self.answer = {}
        self.temp_set = set([])
        self.temp_bool_dict = {}
        self.web_url = 'http://127.0.0.1:5577/post'
        if url is not None:
            self.web_url = url

    def executor(self, throw_exception=False):
        if len(self.seq) > 0:
            for symbolic in self.seq:
                key = list(symbolic.keys())[0]
                if len(symbolic[key]) != 3:
                    continue
                e = symbolic[key][0].strip()
                r = symbolic[key][1].strip()
                t = symbolic[key][2].strip()
                # The execution result from A1 is in dict format.
                # A1: Select(e，r，t)
                if ("A1" in symbolic):
                    try:
                        temp_result = self.select(e, r, t, throw_exception)
                        self.answer = temp_result
                        self.temp_bool_dict = temp_result
                    except Exception as e:
                        print('ERROR! The action is Select(%s,%s,%s).' %(e,r,t))
                        print('The sequence is:')
                        print(self.seq)
                        self.answer = {}
                        if throw_exception:
                            raise e
                    finally:
                        self.print_answer()
                # A2: SelectAll (et, r, t)
                elif ("A2" in symbolic or "A16" in symbolic):
                    try:
                        self.answer = self.select_all(e, r, t, throw_exception)
                    except Exception as e:
                        print('ERROR! The action is SelectAll(%s,%s,%s).' % (e,r,t))
                        print('The sequence is:')
                        print(self.seq)
                        self.answer = {}
                        if throw_exception:
                            raise e
                    finally:
                        self.print_answer()
                # A3: Bool(e)
                elif ("A3" in symbolic):
                    try:
                        if not e:
                            self.answer = {}
                        elif not e.startswith('Q'):
                            self.answer = {}
                        else:
                            bool_temp_result = self.is_bool(e)
                            # dict
                            if type(self.answer) == type({}):
                                if '|BOOL_RESULT|' in self.answer:
                                    self.answer['|BOOL_RESULT|'].append(bool_temp_result)
                                else:
                                    temp = [bool_temp_result]
                                    self.answer.setdefault('|BOOL_RESULT|', temp)
                    except Exception as e:
                        print('ERROR! The action is Bool(%s).' %e)
                        print('The sequence is:')
                        print(self.seq)
                        self.answer = {}
                        if throw_exception:
                            raise e
                    finally:
                        self.print_answer()
                # A4: ArgMin
                elif ("A4" in symbolic):
                    try:
                        self.answer = self.arg_min()
                    except Exception as e:
                        # print('ERROR! The action is ArgMin.')
                        # print('The sequence is:')
                        # print(self.seq)
                        self.answer = {}
                        if throw_exception:
                            raise e
                    finally:
                        self.print_answer()
                # A5: ArgMax
                elif ("A5" in symbolic):
                    try:
                        self.answer = self.arg_max()
                    except Exception as e:
                        # print('ERROR! The action is ArgMax.')
                        # print('The sequence is:')
                        # print(self.seq)
                        self.answer = {}
                        if throw_exception:
                            raise e
                    finally:
                        self.print_answer()
                # A6: GreaterThan(e)
                elif ("A6" in symbolic):
                    try:
                        self.answer = self.greater_than(e,r,t)
                    except Exception as e:
                        print('ERROR! The action is GreaterThan(%s,%s,%s).' % (e,r,t))
                        print('The sequence is:')
                        print(self.seq)
                        self.answer = {}
                        if throw_exception:
                            raise e
                    finally:
                        self.print_answer()
                # A7: LessThan(e)
                elif ("A7" in symbolic):
                    try:
                        self.answer = self.less_than(e,r,t)
                    except Exception as e:
                        # print('ERROR! The action is LessThan(%s,%s,%s).' % (e,r,t))
                        # print('The sequence is:')
                        # print(self.seq)
                        self.answer = {}
                        if throw_exception:
                            raise e
                    finally:
                        self.print_answer()
                # A9: Union(e，r，t)
                elif ("A9" in symbolic):
                    try:
                        self.answer = self.union(e, r, t)
                    except Exception as e:
                        # print('ERROR! The action is Union(%s,%s,%s).' % (e,r,t))
                        # print('The sequence is:')
                        # print(self.seq)
                        self.answer = {}
                        if throw_exception:
                            raise e
                    finally:
                        self.print_answer()
                # A8: Inter(e，r，t)
                elif ("A8" in symbolic):
                    try:
                        self.answer = self.inter(e, r, t)
                    except Exception as e:
                        # print('ERROR! The action is Inter(%s,%s,%s).' % (e,r,t))
                        # print('The sequence is:')
                        # print(self.seq)
                        self.answer = {}
                        if throw_exception:
                            raise e
                    finally:
                        self.print_answer()
                # A10: Diff(e，r，t)
                elif ("A10" in symbolic):
                    try:
                        self.answer = self.diff(e, r, t, throw_exception)
                    except Exception as e:
                        # print('ERROR! The action is Diff(%s,%s,%s).' % (e,r,t))
                        # print('The sequence is:')
                        # print(self.seq)
                        self.answer = {}
                        if throw_exception:
                            raise e
                    finally:
                        self.print_answer()
                # A11: Count(e)
                elif ("A11" in symbolic):
                    try:
                        self.answer = self.count(e)
                    except Exception as e:
                        # print('ERROR! The action is Count(%s).' %e)
                        # print('The sequence is:')
                        # print(self.seq)
                        self.answer = {}
                        if throw_exception:
                            raise e
                    finally:
                        self.print_answer()
                # A12: ATLEAST(N)
                elif ("A12" in symbolic):
                    try:
                        self.answer = self.at_least(e)
                    except Exception as e:
                        # print('ERROR! The action is ATLEAST(%s).' %e)
                        # print('The sequence is:')
                        # print(self.seq)
                        self.answer = {}
                        if throw_exception:
                            raise e
                    finally:
                        self.print_answer()
                # A13: ATMOST(N)
                elif ("A13" in symbolic):
                    try:
                        self.answer = self.at_most(e)
                    except Exception as e:
                        # print('ERROR! The action is ATMOST(%s).' %e)
                        # print('The sequence is:')
                        # print(self.seq)
                        self.answer = {}
                        if throw_exception:
                            raise e
                    finally:
                        self.print_answer()
                # A14: EQUAL(N)
                elif ("A14" in symbolic):
                    try:
                        self.answer = self.equal(e)
                    except Exception as e:
                        # print('ERROR! The action is EQUAL(%s).' %e)
                        # print('The sequence is:')
                        # print(self.seq)
                        self.answer = {}
                        if throw_exception:
                            raise e
                    finally:
                        self.print_answer()
                # A15: around(N)
                elif ("A15" in symbolic):
                    try:
                        if r == "" and t == "":
                            self.answer = self.around(e)
                        else:
                            self.answer = self.around(e,r,t)
                    except Exception as e:
                        if r == "" and t == "":
                            # print('ERROR! The action is Almost(%s).' %e)
                            # print('The sequence is:')
                            # print(self.seq)
                            self.answer = {}
                        else:
                            # print('ERROR! The action is Almost(%s,%s,%s).' %(e,r,t))
                            # print('The sequence is:')
                            # print(self.seq)
                            self.answer = {}
                        if throw_exception:
                            raise e
                    finally:
                        self.print_answer()
                elif ("A17" in symbolic):
                    self.print_answer()
                else:
                    print("wrong symbolic")
        return self.answer

    def is_A(self,e, throw_exception=False):
        #return type of entity
        if e == "":
            return "empty"
        if self.type_dict is not None:
            try:
                return self.type_dict[get_id(e)]
            except:
                return "empty"
        else:
            json_pack = dict()
            json_pack['op']="is_A"
            json_pack['entity']=e
            content = "empty"
            try:
                # content=requests.post("http://127.0.0.1:5000/post",json=json_pack).json()['content']
                content_json = requests.post(self.web_url, json=json_pack).json()
                if 'content' in content_json:
                    content = content_json['content']
            except Exception as e:
                print("ERROR for command: is_A(%s)" %e)
                print('The sequence is:')
                print(self.seq)
                if throw_exception:
                    raise e
            finally:
                return content

    def select(self, e, r, t, throw_exception=False):
        if e == "" or r == "" or t == "":
            return {}
        if self.graph is not None:
            if 'sub' in self.graph[get_id(e)] and r in self.graph[get_id(e)]['sub']:
                return {e:[ee for ee in self.graph[get_id(e)]['sub'][r] if self.is_A(ee) == t]}
            elif 'obj' in self.graph[get_id(e)] and r in self.graph[get_id(e)]['obj']:
                return {e:[ee for ee in self.graph[get_id(e)]['obj'][r] if self.is_A(ee) == t]}
            else:
                return {}
        else:
            json_pack = dict()
            json_pack['op'] = "select"
            json_pack['sub'] = e
            json_pack['pre'] = r
            json_pack['obj'] = t
            content = set([])
            try:
                # content = requests.post("http://127.0.0.1:5000/post", json=json_pack).json()['content']
                content_json = requests.post(self.web_url, json=json_pack).json()
                if 'content' in content_json:
                    content = content_json['content']
            except Exception as e:
                print("ERROR for command: select(%s,%s,%s)" % (e, r, t))
                print('The sequence is:')
                print(self.seq)
                if throw_exception:
                    raise e
            finally:
                if content is not None:
                    # Store records in set.
                    content = set(content)
                else:
                    content = set([])
                # A dict is returned whose key is the subject and whose value is set of entities.
                return {e:content}

    def select_all(self, et, r, t, throw_exception=False):
        #print("A2:", et, r, t)
        content = {}
        if et == "" or r == "" or t == "":
            return content
        if self.graph is not None and self.par_dict is not None:
            keys = self.par_dict[get_id(et)]
            for key in keys:
                if 'sub' in self.graph[get_id(key)] and r in self.graph[get_id(key)]['sub']:
                    content[key] = [ee for ee in self.graph[get_id(key)]['sub'][r] if self.is_A(ee) == t]
                elif 'obj' in self.graph[get_id(key)] and r in self.graph[get_id(key)]['obj']:
                    content[key] = [ee for ee in self.graph[get_id(key)]['obj'][r] if self.is_A(ee) == t]

                else:
                    content[key] = []
            return content
        else:
            json_pack = dict()
            json_pack['op'] = "select_All"
            json_pack['sub'] = et
            json_pack['pre'] = r
            json_pack['obj'] = t
            try:
                content_json = requests.post(self.web_url, json=json_pack).json()
                if 'content' in content_json:
                    content = content_json['content']
            except Exception as  e:
                print("ERROR for command: select_all(%s,%s,%s)" %(et,r,t))
                print('The sequence is:')
                print(self.seq)
                if throw_exception:
                    raise e
            # content = requests.post("http://127.0.0.1:5000/post", json=json_pack).json()['content']
            # for k, v in content.items():
            #   if len(v) == 0: content.pop(k)
            finally:
                if self.answer:
                    for k, v in self.answer.items():
                        # Combine the retrieved entities with existed retrieved entities related to same subject.
                        content.setdefault(k, []).extend(v)
                return content

    def is_bool(self, e):
        # print("A3: is_bool")
        if type(self.answer) == bool: return self.answer
        if self.temp_bool_dict == None: return False
        if type(self.temp_bool_dict) == dict:
            for key in self.temp_bool_dict:
                if (self.temp_bool_dict[key] != None and e in self.temp_bool_dict[key]):
                    return True
        return False


    def arg_min(self):
        # print("A4: arg_min")
        if not self.answer:
            return []
        if type(self.answer) != dict: return []
        minK = min(self.answer, key=lambda x: len(self.answer[x]))
        minN = len(self.answer[minK])
        min_set = [k for k in self.answer if len(self.answer[k]) == minN]
        self.temp_set = set(min_set)
        return min_set


    def arg_max(self):
        # print("A5: arg_max")
        if not self.answer:
            return []
        if type(self.answer) != dict: return []
        maxK = max(self.answer, key=lambda x: len(self.answer[x]))
        maxN = len(self.answer[maxK])
        return [k for k in self.answer if len(self.answer[k]) == maxN]


    def greater_than(self, e, r, t):
        content = self.answer
        if type(content) != dict: return []
        if e in content and not content[e] == None:
            N = len(content[e])
        else:
            N = 0
        return [k for k in self.answer if len(self.answer[k]) > N]

    def less_than(self, e, r, t):
        content = self.answer
        if type(content) != dict: return []
        if e in content and not content[e] == None:
            N = len(content[e])
        else:
            N = 0
        return [k for k in self.answer if len(self.answer[k]) < N]


    def union(self, e, r, t, throw_exception=False):
        #print("A9:", e, r, t)
        if e == "": return {}
        if not e.startswith("Q"): return {}
        answer_dict = self.answer
        if type(answer_dict) == bool: return False
        elif type(answer_dict) != dict: return {}
        try:
            if e in answer_dict and answer_dict[e]!=None:
                temp_dict = self.select(e, r, t)
                if e in temp_dict:
                    answer_dict[e] = set(answer_dict[e]) | set(temp_dict[e])
            else:
                answer_dict.update(self.select(e, r, t))
        except Exception as e:
            print("ERROR for command: union(%s,%s,%s)" % (e, r, t))
            print('The sequence is:')
            print(self.seq)
            if throw_exception:
                raise e
        finally:
            union_key = "|"
            union_value = set([])
            for k, v in answer_dict.items():
                if v == None: v = []
                union_value = union_value | set(v)
            answer_dict.clear()
            answer_dict[union_key] = list(set(union_value))
            return answer_dict


    def inter(self, e, r, t, throw_exception=False):
        #print("A8:", e, r, t)
        if e == "": return {}
        if not e.startswith("Q"): return {}
        answer_dict = self.answer
        if type(answer_dict) != dict: return {}
        try:
            if e in answer_dict and answer_dict[e]!=None:
                temp_dict = self.select(e, r, t)
                if e in temp_dict:
                    answer_dict[e] = set(answer_dict[e]) & set(temp_dict[e])
            else:   # ??????
                s = self.select(e, r, t)
                answer_dict.update(s)
        except Exception as e:
            print("ERROR for command: inter(%s,%s,%s)" % (e, r, t))
            print('The sequence is:')
            print(self.seq)
            if throw_exception:
                raise e
        finally:
            inter_key = "&"  # there is a bug in the origin code, [entity1] & [] will be [] while [] & [entity1] will be [entity1]
            inter_value = set([])
            # for k, v in answer_dict.items():
            #     if v == None: v = []
            #     if len(inter_value) > 0:
            #         inter_value = inter_value & set(v)
            #     else:
            #         inter_value = set(v)
            for k, v in answer_dict.items():
                if v == None: v = []
                if len(v) == 0:
                    inter_value = set([])
                    break
                if len(inter_value) > 0:
                    inter_value = inter_value & set(v)
                else:
                    inter_value = set(v)
            answer_dict.clear()
            answer_dict[inter_key] = list(set(inter_value))
            return answer_dict

    def diff(self, e, r, t, throw_exception=False):
        #print("A10:", e, r, t)
        if e == "": return {}
        if not e.startswith("Q"): return {}
        answer_dict = self.answer
        if type(answer_dict) != dict: return {}
        try:
            if e in answer_dict and answer_dict[e]!=None:
                temp_dict = self.select(e, r, t)
                if e in temp_dict:
                    answer_dict[e] = set(answer_dict[e]) - set(temp_dict[e])
            else:
                answer_dict.update(self.select(e, r, t))
        except Exception as e:
            print("ERROR for command: diff(%s,%s,%s)" % (e, r, t))
            print('The sequence is:')
            print(self.seq)
            if throw_exception:
                raise e
        finally:
            diff_key = "-"
            diff_value = set([])
            for k, v in answer_dict.items():
                if v == None: v = []
                if k != e:
                    diff_value.update(set(v))
            if(answer_dict[e]):
                diff_value = diff_value - set(answer_dict[e])
            answer_dict.clear()
            answer_dict[diff_key] = list(set(diff_value))
            return answer_dict

    def count(self,e= None):
        #print("A11:Count")
        try:
            # list or set
            if type(self.answer) == type([]) or type(self.answer) == type(set()):
                return len(self.answer)
            # dict
            if type(self.answer) == type({}):
                if e!='' and e:
                    if e not in self.answer and len(self.answer.keys()) == 1:
                        return len(self.answer.popitem())
                    elif e in self.answer:
                        return len(self.answer[e])
                else:
                    if len(self.answer.keys()) == 1:
                        return len(self.answer.popitem()[1])
                    return len(self.answer.keys())
            # int
            if type(self.answer) == type(1):
                return self.answer
            else:
                return 0
        except:
            # print("ERROR! THE ACTION IS count(%s)!" %e)
            # print('The sequence is:')
            # print(self.seq)
            return 0

    def at_least(self, N):
        # print("A12: at_least")
        # for k in list(self.answer):
        #     if len(self.answer[k]) <= int(N):
        #         self.answer.pop(k)
        # return self.answer
        answer_keys = []
        if type(self.answer) == dict:
            for k, v in self.answer.items():
                if len(v) >= int(N):
                    answer_keys.append(k)
        return answer_keys

    def at_most(self, N):
        # print("A13: at_most")
        answer_keys = []
        # for k, v in self.answer.items():
        #   if len(v) == 0: self.answer.pop(k)
        if type(self.answer) == dict:
            for k in list(self.answer):
                if len(self.answer[k]) <= int(N):
                    answer_keys.append(k)
        return answer_keys

    def equal(self, N):
        answer_keys = []
        if type(self.answer) == dict:
            for k, v in self.answer.items():
                #print k,len(v)
                if len(v) == int(N):
                    answer_keys.append(k)
        return answer_keys

    def around(self,N,r=None,t=None, throw_exception=False):
        answer_keys = []
        number = 0
        try:
            if N.isdigit():
                number = int(N)
            elif N.startswith("Q"):
                if(r!=None) and (t!=None):
                    e = N
                    dict_temp = self.select_all(e,r,t)
                    number = len(dict_temp)
                else:
                    content = self.answer
                    if type(content) == dict:
                        if N in content and not content[N] == None:
                            number = len(content[N])
                        else:
                            number = 0
            # print(sorted([(k, len(self.answer[k])) for k in self.answer.keys() if len(self.answer[k]) > 0], key=lambda x: x[1]))
            # If N is not digit nor started with 'Q', the number is assumed as 0.
            if type(self.answer) == type({}):
                if number == 0 or 1< number <=5:
                    for k, v in self.answer.items():
                        if abs(len(v)-int(number)) <= 1:
                            answer_keys.append(k)
                elif number == 1:
                    for k, v in self.answer.items():
                        if abs(len(v) - int(number)) < (int(number) * 0.6):
                            answer_keys.append(k)
                elif number > 5:
                    for k, v in self.answer.items():
                        if abs(len(v)-int(number)) <= 5:  
                            answer_keys.append(k)
                else:
                    for k, v in self.answer.items():
                        # print k, len(v),abs(len(v)-int(N)),(int(N)/2)
                        if abs(len(v)-int(number)) < (int(number)*0.6):
                            answer_keys.append(k)
                self.temp_set = set(answer_keys)
        except Exception as e:
            print('ERROR!The action is around(%s, %s, %s)' %(N,r,t))
            print('The sequence is:')
            print(self.seq)
            if throw_exception:
                raise e
        finally:
            return answer_keys

    def EOQ(self):
        pass

    ########################
    def print_answer(self):
        pass
        # if(type(self.answer) == dict):
        #     for k,v in self.answer.items():
        #         #print self.item_data[k],": ",
        #         for value in v:
        #         #    print self.item_data[value], ",",
        #         print
        # elif(type(self.answer) == type([])):
        #     for a in self.answer:
        #         print self.item_data[a],
        #     print
        # else:
        #     if(self.answer in self.item_data):
        #         print self.answer,self.item_data[self.answer]
        #     else:
        #         print self.answer
    # print("----------------")

    def select_sparql(self, e, r, t):  # use sparql
        answer_dict = {}
        anser_values = []
        sparql = {"query": "SELECT ?river WHERE { \
                                            ?river wdt:" + r + " wd:" + e + ". \
                                            ?river wdt:P31  wd:" + t + ". \
                                       }",
                  "format": "json",
                  }
        # print sparql
        sparql = urlencode(sparql)
        # print sparql
        url = 'https://query.wikidata.org/sparql?' + sparql
        r = requests.get(url)
        # print r.json()["results"]
        for e in r.json()["results"]["bindings"]:
            entity = e["river"]["value"].split("/")[-1]
            anser_values.append(entity)
        answer_dict[e] = anser_values

        return answer_dict

if __name__ == "__main__":
    print("Building knowledge base....")
    d =  [{'A2': ['Q502895', '-P417', 'Q12737077']}, {'A2': ['Q502895', '-P417', 'Q502895']}, {'A15': ['Q193507', '', '']}, {'A11': ['Q193507', '', '']}]
    symbolic_exe = Symbolics(d)
    answer1 = symbolic_exe.executor()
    print(answer1)
