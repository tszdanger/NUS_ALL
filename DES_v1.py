

'''
测试样例：
明文：1000011110000111100001111000011110000111100001111000100000000000
0000000100100011010001010110011110001001101010111100110111101111

1000010111101000000100110101010000001111000010101011010000000101
[1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]

[1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
密钥：0000111000110010100100100011001011101010011011010000110110000000
 0001001100110100010101110111100110011011101111001101111111110001
[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
'''
# 初始置换(正确的)
# 输入：64位二进制数据(默认补00放到数据处理里面做)[1,0,0,1...0,1]
# 输出: res(L0,R0)
def inial_displace_data(data_64):
    a = [58, 50, 42, 34, 26, 18, 10, 2,
         60, 52, 44, 36, 28, 20, 12, 4,
         62, 54, 46, 38, 30, 22, 14, 6,
         64, 56, 48, 40, 32, 24, 16, 8,
         57, 49, 41, 33, 25, 17, 9, 1,
         59, 51, 43, 35, 27, 19, 11, 3,
         61, 53, 45, 37, 29, 21, 13, 5,
         63, 55, 47, 39, 31, 23, 15, 7, ]
    res = []
    for i in a:
        res.append(data_64[i-1])


    return res
#初始密钥置换(正确已验证)
#input:key_64
#output:res(c0,d0)56位

def inial_displace_key(key_64):
    a = [57, 49, 41, 33, 25, 17, 9, 1,
         58, 50, 42, 34, 26, 18, 10, 2,
         59, 51, 43, 35, 27, 19, 11, 3,
         60, 52, 44, 36, 63, 55, 47, 39,
         31, 23, 15, 7, 62, 54, 46, 38,
         30, 22, 14, 6, 61, 53, 45, 37,
         29, 21, 13, 5, 28, 20, 12, 4]
    res = []
    for i in a:
        res.append(key_64[i-1])#这个表里面的意思是去除第八位
    return res
#num_shift = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]16次移位
#有效密钥移位(没问题)
#input:56位密钥,移位次数
#output:子密钥

def shift_key(key_56,shift_num):
    new_key1 = key_56[:28]
    new_key2 = key_56[28:]
    if(shift_num>=28):
        shift_num -=28
    c0 = new_key1[shift_num:]
    c0 += new_key1[:shift_num]
    c1 = new_key2[shift_num:]
    c1 += new_key2[:shift_num]
    # print(c0)
    # print(c1)
    new_key = c0+c1
    return new_key

#对子密钥进行压缩得到48位的(没有问题)
#input: 56位密钥
#output: 48位密钥
def  compress_key(key_56):
    a = [14, 17, 11, 24, 1, 5,
         3, 28, 15, 6, 21, 10,
         23, 19, 12, 4, 26, 8,
         16, 7, 27, 20, 13, 2,
         41, 52, 31, 37, 47, 55,
         30, 40, 51, 45, 33, 48,
         44, 49, 39, 56, 34, 53,
         46, 42, 50, 36, 29, 32]
    res = []
    for i in a:
        res.append(key_56[i-1])
    # print('48位压缩密钥')
    # print(res)
    return res

#对数据的L0不变，L1扩展为48位以后做异或
#input: key_48,data_64
#output: encro_48 已经加密的48位数据
def xorkey_data(data_64,key_48):
    L = data_64[:32]
    R = data_64[32:]
    a =[32, 1, 2, 3, 4, 5,
        4, 5, 6, 7, 8, 9,
        8, 9, 10, 11, 12, 13,
        12, 13, 14, 15, 16, 17,
        16, 17, 18, 19, 20, 21,
        20, 21, 22, 23, 24, 25,
        24, 25, 26, 27, 28, 29,
        28, 29, 30, 31, 32, 1]
    R_48 = [] #扩展后的R
    for i in a:
        R_48.append(R[i-1])

    res = []
    for i in range(len(R_48)):
        res.append(R_48[i]^key_48[i])

    return res
#给定一个六位的[],返回两个数字代表行和列
def six_to_four(six_data):
    res = []
    str1= str(six_data[0])+str(six_data[5])
    res.append(int(str1,2))#行
    str2 = str(six_data[1])+str(six_data[2])+str(six_data[3])+str(six_data[4])
    res.append(int(str2,2))#列
    return  res
#给一个48位的数据，返回8个六位的.[[1,1,1,1,1,1],[...],...]
def eight_six(encry_data_48):
    count = 0
    temp_list = []
    res = []
    for i in encry_data_48:
        temp_list.append(i)
        count +=1
        if (count==6):
            res.append(temp_list)
            count = 0
            temp_list=[]
    return res
#给一个十进制数字，返回位数固定为4位的二进制数
#input: 0-15,output: 0000-1111
def int_to4bin(num):
    bin_num = bin(num)
    str1 = bin_num[2:]
    length = 4-len(str1)#需要补充的位数
    res = '0'*length+str1
    return res

#S盒加密算法(正确)
#input:48位加密的数据
#output: 32位加密的数组
def new_32(encry_data_48):
    s1 = [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
          0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
          4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
          15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13, ]
    s2 = [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
          3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
          0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
          13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9, ]
    s3 = [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
          13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
          13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
          1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12, ]
    s4 = [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
          13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
          10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
          3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14, ]
    s5 = [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
          14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
          4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
          11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3, ]
    s6 = [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
          10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
          9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
          4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13, ]
    s7 = [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
          13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
          1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
          6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12, ]
    s8 = [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
          1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
          7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
          2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11, ]
    dic_chooses = {0:s1,1:s2,2:s3,3:s4,4:s5,5:s6,6:s7,7:s8}
    temp_data = eight_six(encry_data_48)
    #temp_data =[[6位],[]...总共8个]
    count1 = 0
    res = []#最终返回的32位数据
    for i in temp_data:
        rowcol = six_to_four(i)
        row = rowcol[0]
        col = rowcol[1]
        new_4 = dic_chooses[count1][row*16+col]   #返回的是一个十进制的值
        k = int_to4bin(new_4)
        for j in k:
            res.append(int(j))
        count1 +=1
    return res

#p盒加密算法(正确)
#input 32位数据
#output: 32位数据(置换后)
def p_box(data_32):
    a = [16, 7, 20, 21, 29, 12, 28, 17, 1, 15, 23, 26, 5, 18, 31, 10,
         2, 8, 24, 14, 32, 27, 3, 9, 19, 13, 30, 6, 22, 11, 4, 25, ]
    res = []
    for i in a:
        res.append(data_32[i-1])
    return res

#最终密文置换
def fianl_dis(data_64):
    a= [40, 8, 48, 16, 56, 24, 64, 32, 39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30, 37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28, 35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26, 33, 1, 41, 9, 49, 17, 57, 25, ]
    res = []
    for i in a:
        res.append(data_64[i-1])
    return res

def data_in():
    # data = bin(int(input('请输入要加密的数据:(v1仅支持数字)')))
    # length1 = 64+2- len(data)
    # res =[0]*length1
    # for i in range(len(data)-2):
    #     res.append(int(data[i+2]))
    # return res
    data = ((input('请输入要加密的data:(v1仅支持数字)')))

    # length1 = 64 + 2 - len(key)
    # res = [0] * length1
    # for i in range(len(key) - 2):
    #     res.append(int(key[i + 2]))
    length1 = 64 - len(data)
    res = [0] * length1
    for i in range(len(data)):
        res.append(int(data[i]))
    return res

def key_in():
    # key = bin(int(input('请输入要加密的密钥:(v1仅支持数字)')))
    key = ((input('请输入要加密的密钥:(v1仅支持数字)')))

    # length1 = 64 + 2 - len(key)
    # res = [0] * length1
    # for i in range(len(key) - 2):
    #     res.append(int(key[i + 2]))
    length1 = 64  - len(key)
    res = [0] * length1
    for i in range(len(key) ):
        res.append(int(key[i]))
    return res


if __name__ == '__main__':
    data_in = data_in()
    key_in = key_in()
    data_turn = inial_displace_data(data_in) #64位初始化数据
    key_turn = inial_displace_key(key_in) #56位初始化密钥
    num_shift = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]
    num_rev_shift = [28, 27, 25, 23, 21, 19, 17, 15, 14, 12, 10, 8, 6, 4, 2, 1]
    for i in range(16):  #进行16轮
        key_turn = shift_key(key_turn,num_shift[i])
        key_48 = compress_key(key_turn)
        encro_48 = xorkey_data(data_turn,key_48)
        R0 = new_32(encro_48)
        RR0 = p_box(R0)
        temp = data_turn
        data_turn = data_turn[32:]
        for j in range(32):
            data_turn.append(temp[j]^RR0[j])
        if(i==15):
            L16 = data_turn[:32]
            R16 = data_turn[32:]
            data_turn = R16+L16

    # 011000010001011110111010100001100110010100100111
    # [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1]
    cro = fianl_dis(data_turn)
    print('最终密文为:')
    print(cro)
    #因为我们已经有了密文和密钥，现在来做解密
    data_turn = inial_displace_data(cro)

    key_turn = inial_displace_key(key_in)
    for i in range(16):
        key_this = shift_key(key_turn,num_rev_shift[i])
        key_48 = compress_key(key_this)
        encro_48 = xorkey_data(data_turn, key_48)
        R0 = new_32(encro_48)
        RR0 = p_box(R0)
        temp = data_turn
        data_turn = data_turn[32:]
        for j in range(32):
            data_turn.append(temp[j] ^ RR0[j])
        if (i == 15):
            L16 = data_turn[:32]
            R16 = data_turn[32:]
            data_turn = R16 + L16

    real = fianl_dis(data_turn)
    print('反解的明文是:')
    print(real)