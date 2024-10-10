# 一级标题

## 二级标题

### 三级标题

###### 六级标题

###### #七级标题

最多六级标题，再多了就识别不了了

`**加粗**`
`两个反引号里面写代码不会被渲染出来`

`*斜体*`

**加粗**

*斜体*

# 列表（有序or无序）

1. 第一点（1.后面加一个空格接着回车就会自动排序）
2. 第二点
3. 第三点
   1. 第3.1点（回车到这一行按一下tab键即可）
   2. 此处想要跳出来就需要连续两次按回车，在下一点会演示
4. 我是你

`无序列表是 *+空格 回车会自动填充新的点`

* 我是你
* 你是我



# 表格

markdown的表格非常简陋，基本就是几行几列（latex无所不能）

`|内容|内容|`

| 阿   | 亮   |
| ---- | ---- |
|      |      |

| head1 | head2 | head3 |
| :---: | :---: | ----- |
|       |       |       |

| :----- | -----: | :--: |
| ------ | ------ | ---- |
|        |        |      |



`***为一条横线`

***



这里注意需要多行表格，要进入源代码模式进行书写，接着退出源代码模式就可以生成多行自定义表格了。

在实际使用中常用到表格相关的快捷键，在此一一列出：

Ctrl + T, 插入新表格

Ctrl + Enter, 在下方插入新行

Ctrl + Shift + Backspace, 删除当前行

| head1 | head2 | head3|
|:--       | --:         | :--:  |
|              |             |           |
|              |             |           |
| | | |
| | | |



# 数学公式

https://www.zybuluo.com/codeep/note/163962

## 行内公式（嵌入在文本内）



`链接：Typora如何插入行内公式（https://www.onlinedown.net/article/10028879.htm）`

`初次使用需要点击文件，偏好设置，markdown，高亮和内联公式都打上对钩，接着重启typora即可使用行内公式`

`$\sum_i^k$`

$1$

$\sum_i^k$

## 行外公式(一般是单独一行)

$$
11
$$

```basic
``` 三个反引号，里面的内容可以换行
$$
数学公式
$$

print('hello world')

```

`\tag {行标}可以用来修改公式排号（自动标号在文件，偏好设置，找到公式，自动添加行号即可）`

公式如下：
$$
\begin{aligned}
x & = \sum_{i=1}^k=\frac{x_1}{h_2} \\  
& =y + 1
\end{aligned}
$$



$$
\begin{aligned}
x & = \sum_{i=1}^k=\frac{x_1}{h_2} \\  
& =y + 1
\end{aligned}
\tag {这儿的行号可以自定义}
$$

### 数学公式测试



https://www.zybuluo.com/codeep/note/163962#1%E5%A6%82%E4%BD%95%E6%8F%92%E5%85%A5%E5%85%AC%E5%BC%8F

$J_\alpha(x)$

$ J_\alpha(x) = \sum_{m=0}^\infty \frac{(-1)^m}{m! \Gamma (m + \alpha + 1)} {\left({ \frac{x}{2} }\right)}^{2m + \alpha} \text {，行内公式示例} $

`\tag在这个中没有起到编号的作用`

$$ J_\alpha(x) = \sum_{m=0}^\infty \frac{(-1)^m}{m! \Gamma (m + \alpha + 1)} {\left({ \frac{x}{2} }\right)}^{2m + \alpha} \text{，独立公式示例} \tag{1}  $$
$$
\begin{equation}
    E=mc^2 \text{，自动编号公式示例}
    \label{eq:sample}
\end{equation}
$$
`不想要加编号，就在begin和end末尾加 * `
$$
\begin{equation*}
    表达式,不加编号
\end{equation*}
$$
2.上下标以及左右两边都有上下标：

$$ x^{y^z}=(1+{\rm e}^x)^{-2xy^w} $$

$$ \sideset{^1_2}{^3_4}\bigotimes \quad or \quad {^1_2}\bigotimes {^3_4} $$

`$\bigotimes$`就是这个符号$\bigotimes$

其余用到啥学啥

# 图片和超链接



https://i2.hdslb.com/bfs/face/0c18d037b0852c384f1306d212b01b5a14796c51.jpg@240w_240h_1c_1s.webp

<img class="bili-avatar-img bili-avatar-face bili-avatar-img-radius" data-src="https://i2.hdslb.com/bfs/face/0c18d037b0852c384f1306d212b01b5a14796c51.jpg@240w_240h_1c_1s.webp" alt="" src="https://i2.hdslb.com/bfs/face/0c18d037b0852c384f1306d212b01b5a14796c51.jpg@240w_240h_1c_1s.webp">





`![](图片地址即可，记住是仅剩jpg或png的)`

`https://i2.hdslb.com/bfs/face/0c18d037b0852c384f1306d212b01b5a14796c51.jpg`

<img src="https://i2.hdslb.com/bfs/face/0c18d037b0852c384f1306d212b01b5a14796c51.jpg" style="zoom:33%;" />





`[](链接地址即可)，接着 ctrl+鼠标左键单击即可`

[图片](https://i2.hdslb.com/bfs/face/0c18d037b0852c384f1306d212b01b5a14796c51.jpg)

[markdown数学公式](https://www.zybuluo.com/codeep/note/163962)

# 分栏符

`--- 即为分栏符`

---

引入 `>`

> 文本引入
>
> 参考文献
>
> 1.1 #####



# 补充(Typora默认不支持，需要自己启动，包括数学公式)

注意：只有Typora支持，其他md文本编辑器不一定支持

`~2~是把1当做下标  ^0^是把0当做上标，==高亮==是高亮，具体如下`

H~2~^0^

==高亮==



# 想要回车没有空行，那就需要按shift+enter

11

11
11



# 最重要的来了

文件--》导出

可以导出为各种文件





# 纸上得来终觉浅，绝知此事要躬行



![image-20240911211046707](./assets/image-20240911211046707.png)
