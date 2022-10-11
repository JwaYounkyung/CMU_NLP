#lemmatizer model solution

# TODO: Consonants subject to the doubling rule
DCONSONANTS = ['p', 't', 'g', 'm', 's', 'n', 'f', 'z', 'd']

# Morphotactic wFST
MORPHOTACTIC = '''
0 1 <oth> <oth> 0.0
1 2 # # 0.3
1 1 <oth> <oth> 1.0
2 0
1 3 <eps> ^ 0.0
3 4 <eps> i 0.0
4 5 <eps> n 0.0
5 6 <eps> g 0.0
6 7 # # 0.0
7 0.0
3 8 <eps> e 0.0
8 9 <eps> d 0.0
9 7 # # 0.0
3 10 <eps> s 0.0
10 7 # # 0.0
'''

# TODO: Silent E-Deletion wFST
E_DELETION = '''
0 1 <oth> <oth> 0.0
1 7 # # 0.3
1 1 <oth> <oth> 0.1
1 1 e e 0.0
1 1 ^ ^ 0.0
1 2 e <eps> 0.0
2 3 ^ ^ 0.0
3 4 i i 0.0
4 5 n n 0.0
5 6 g g 0.0
6 7 # # 0.0
7 0.0
3 8 e e 0.0
8 9 d d 0.0
9 7 # # 0.0
'''

# TODO: Silent E-Insertion wFST
E_INSERTION = '''
0 0 <oth> <oth> 0.1
0 0 <c> <c> 0.0
0 0 ^ ^ 0.0
0 0 e e 0.0
0 1 s s 0.0
0 2 z z 0.0
0 3 x x 0.0
0 4 s s 0.0
4 5 h h 0.0
0 6 c c 0.0
6 7 h h 0.0
0 11 # # 0.3
11 0.0
1 8 ^ ^ 0.0
2 8 ^ ^ 0.0
3 8 ^ ^ 0.0
5 8 ^ ^ 0.0
7 8 ^ ^ 0.0
8 9 <eps> e 0.0
9 10 s s 0.0
10 11 # # 0.0
'''

# Y-Replacement wFST
Y_REPLACEMENT = """
0 0 <oth> <oth> 0.1
0 0 <c> <c> 0.0
0 0 ^ ^ 0.0
0 0 y y 0.1
0 8 <c> <c> 0.0
8 1 y i 0.0
0 7 # # 1.0
1 2 ^ ^ 0.0
2 3 <eps> e 0.0
2 5 e e 0.0
3 4 s s 0.01
4 7 # # 0.0
5 6 d d 0.0
6 7 # # 0.0
7 0.0"""