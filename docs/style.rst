


Terminology
-----------

+-------------+----------------------------------------------+------+--------------+
|Name         |Definition                                    | Type | Class        |
+=============+==============================================+======+==============+
|spacer       |The (usually) 20 bp RNA segment of the gRNA   | RNA  | Bio.Seq      |
|             |which directs Cas9 to the intended target     |      |              |
+-------------+----------------------------------------------+------+--------------+
|target       |The region of the DNA which is targeted by    | DNA  | Bio.Seq      |
|             |the Cas9 for binding. This segment is the     |      |              |
|             |DNA strand that is homologous to the `spacer` |      |              |
|             |and includes the PAM sequence.                |      |              |
+-------------+----------------------------------------------+------+--------------+
|locus        |The region of DNA against which `spacers` are | DNA  | Bio.SeqRecord|
|             |designed. This is typically a DNA sequence    |      |              |
|             |longer than the 23 bp target.                 |      |              |
+-------------+----------------------------------------------+------+--------------+