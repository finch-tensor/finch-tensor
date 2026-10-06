Compiling MLIR code:
module {
  func.func @main(%_A_15: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>, %_A_16: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>, %_A_18: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_A_7: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_15[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_2 = llvm.getelementptr %v[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_3 = llvm.load %v_2 : !llvm.ptr -> !llvm.ptr
    %v_4 = llvm.getelementptr %v[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_5 = llvm.load %v_4 : !llvm.ptr -> i64
    %v_6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_7 = arith.constant 0 : i64
    %v_8 = arith.constant 1 : i64
    %v_9 = llvm.insertvalue %v_3, %v_6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_10 = llvm.insertvalue %v_3, %v_9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_11 = llvm.insertvalue %v_7, %v_10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_12 = llvm.insertvalue %v_5, %v_11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_13 = llvm.insertvalue %v_8, %v_12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_14 = builtin.unrealized_conversion_cast %v_13 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_15 = builtin.unrealized_conversion_cast %v_14 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_16 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_15, %v_16 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_17 = llvm.extractvalue %_A_15[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_18 = arith.index_cast %v_17 : i64 to index
    %v_19 = llvm.extractvalue %_A_15[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_20 = arith.index_cast %v_19 : i64 to index
    %v_21 = llvm.extractvalue %_A_16[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_22 = llvm.getelementptr %v_21[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_23 = llvm.load %v_22 : !llvm.ptr -> !llvm.ptr
    %v_24 = llvm.getelementptr %v_21[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_25 = llvm.load %v_24 : !llvm.ptr -> i64
    %v_26 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_27 = llvm.insertvalue %v_23, %v_26[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_28 = llvm.insertvalue %v_23, %v_27[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_29 = llvm.insertvalue %v_7, %v_28[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_30 = llvm.insertvalue %v_25, %v_29[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_31 = llvm.insertvalue %v_8, %v_30[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_32 = builtin.unrealized_conversion_cast %v_31 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_33 = builtin.unrealized_conversion_cast %v_32 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_34 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_33, %v_34 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_35 = llvm.extractvalue %_A_16[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_36 = arith.index_cast %v_35 : i64 to index
    %v_37 = llvm.extractvalue %_A_16[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_38 = arith.index_cast %v_37 : i64 to index
    %v_39 = llvm.extractvalue %_A_18[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_40 = llvm.getelementptr %v_39[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_41 = llvm.load %v_40 : !llvm.ptr -> !llvm.ptr
    %v_42 = llvm.getelementptr %v_39[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_43 = llvm.load %v_42 : !llvm.ptr -> i64
    %v_44 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_45 = llvm.insertvalue %v_41, %v_44[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_46 = llvm.insertvalue %v_41, %v_45[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_47 = llvm.insertvalue %v_7, %v_46[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_48 = llvm.insertvalue %v_43, %v_47[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_49 = llvm.insertvalue %v_8, %v_48[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_50 = builtin.unrealized_conversion_cast %v_49 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_51 = builtin.unrealized_conversion_cast %v_50 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_52 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_51, %v_52 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_53 = llvm.extractvalue %_A_18[1, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_54 = arith.index_cast %v_53 : i64 to index
    %v_55 = llvm.extractvalue %_A_18[1, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_56 = arith.index_cast %v_55 : i64 to index
    %v_57 = llvm.extractvalue %_A_7[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_58 = llvm.getelementptr %v_57[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_59 = llvm.load %v_58 : !llvm.ptr -> !llvm.ptr
    %v_60 = llvm.getelementptr %v_57[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_61 = llvm.load %v_60 : !llvm.ptr -> i64
    %v_62 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_63 = llvm.insertvalue %v_59, %v_62[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_64 = llvm.insertvalue %v_59, %v_63[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_65 = llvm.insertvalue %v_7, %v_64[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_66 = llvm.insertvalue %v_61, %v_65[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_67 = llvm.insertvalue %v_8, %v_66[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_68 = builtin.unrealized_conversion_cast %v_67 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_69 = builtin.unrealized_conversion_cast %v_68 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_70 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_69, %v_70 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_71 = llvm.extractvalue %_A_7[1, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_72 = arith.index_cast %v_71 : i64 to index
    %v_73 = llvm.extractvalue %_A_7[1, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_74 = arith.index_cast %v_73 : i64 to index
    %v_75 = arith.constant 0 : index
    %v_76 = llvm.load %v_52 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_77 = builtin.unrealized_conversion_cast %v_76 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_78 = memref.dim %v_77, %v_75 : memref<?xf64>
    %v_79 = arith.constant 1 : index
    scf.for %v_80 = %v_75 to %v_78 step %v_79 {
      %v_81 = llvm.load %v_52 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_82 = builtin.unrealized_conversion_cast %v_81 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_83 = arith.constant 0.0 : f64
      memref.store %v_83, %v_82[%v_80] : memref<?xf64>
    }
    scf.for %v_84 = %v_75 to %v_20 step %v_79 {
      %v_85 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_86 = arith.index_cast %v_85 : i64 to index
      %v_87 = arith.muli %v_86, %v_84 : index
      scf.for %v_88 = %v_75 to %v_18 step %v_79 {
        %v_89 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_90 = arith.index_cast %v_89 : i64 to index
        %v_91 = arith.muli %v_90, %v_88 : index
        %v_92 = arith.addi %v_87, %v_91 : index
        scf.for %v_93 = %v_75 to %v_20 step %v_79 {
          %v_94 = arith.muli %v_88, %v_20 : index
          %v_95 = arith.addi %v_94, %v_93 : index
          %v_96 = arith.cmpi eq, %v_93, %v_84 : index
          scf.if %v_96 {
            %v_97 = llvm.load %v_52 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_98 = builtin.unrealized_conversion_cast %v_97 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_99 = llvm.load %v_16 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_100 = builtin.unrealized_conversion_cast %v_99 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_101 = memref.load %v_100[%v_95] : memref<?xf64>
            memref.store %v_101, %v_98[%v_92] : memref<?xf64>
          }
        }
      }
    }
    %v_102 = llvm.load %v_70 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_103 = builtin.unrealized_conversion_cast %v_102 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_104 = memref.dim %v_103, %v_75 : memref<?xf64>
    scf.for %v_105 = %v_75 to %v_104 step %v_79 {
      %v_106 = llvm.load %v_70 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_107 = builtin.unrealized_conversion_cast %v_106 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_108 = arith.constant 0.0 : f64
      memref.store %v_108, %v_107[%v_105] : memref<?xf64>
    }
    scf.for %v_109 = %v_75 to %v_36 step %v_79 {
      %v_110 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_111 = arith.index_cast %v_110 : i64 to index
      %v_112 = arith.muli %v_111, %v_109 : index
      scf.for %v_113 = %v_75 to %v_56 step %v_79 {
        %v_114 = llvm.extractvalue %_A_7[2, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_115 = arith.index_cast %v_114 : i64 to index
        %v_116 = arith.muli %v_115, %v_113 : index
        %v_117 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_118 = arith.index_cast %v_117 : i64 to index
        %v_119 = arith.muli %v_118, %v_113 : index
        %v_120 = arith.addi %v_112, %v_119 : index
        scf.for %v_121 = %v_75 to %v_38 step %v_79 {
          %v_122 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_123 = arith.index_cast %v_122 : i64 to index
          %v_124 = arith.muli %v_123, %v_121 : index
          %v_125 = arith.addi %v_116, %v_124 : index
          %v_126 = arith.muli %v_109, %v_38 : index
          %v_127 = arith.addi %v_126, %v_121 : index
          %v_128 = llvm.load %v_70 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_129 = builtin.unrealized_conversion_cast %v_128 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_130 = llvm.load %v_70 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_131 = builtin.unrealized_conversion_cast %v_130 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_132 = memref.load %v_131[%v_125] : memref<?xf64>
          %v_133 = llvm.load %v_52 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_134 = builtin.unrealized_conversion_cast %v_133 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_135 = memref.load %v_134[%v_120] : memref<?xf64>
          %v_136 = llvm.load %v_34 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_137 = builtin.unrealized_conversion_cast %v_136 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_138 = memref.load %v_137[%v_127] : memref<?xf64>
          %v_139 = arith.mulf %v_135, %v_138 : f64
          %v_140 = arith.addf %v_132, %v_139 : f64
          memref.store %v_140, %v_129[%v_125] : memref<?xf64>
        }
      }
    }
    %v_141 = llvm.load %v_16 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_142 = builtin.unrealized_conversion_cast %v_141 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_143 = llvm.load %v_16 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_144 = builtin.unrealized_conversion_cast %v_143 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_145 = memref.dim %v_144, %v_75 : memref<?xf64>
    %v_146 = arith.index_cast %v_145 : index to i64
    %v_147 = llvm.getelementptr %v[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_146, %v_147 : i64, !llvm.ptr
    %v_148 = memref.extract_aligned_pointer_as_index %v_142 : memref<?xf64> -> index
    %v_149 = arith.index_cast %v_148 : index to i64
    %v_150 = llvm.inttoptr %v_149 : i64 to !llvm.ptr
    %v_151 = llvm.getelementptr %v[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_150, %v_151 : !llvm.ptr, !llvm.ptr
    %v_152 = llvm.load %v_34 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_153 = builtin.unrealized_conversion_cast %v_152 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_154 = llvm.load %v_34 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_155 = builtin.unrealized_conversion_cast %v_154 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_156 = memref.dim %v_155, %v_75 : memref<?xf64>
    %v_157 = arith.index_cast %v_156 : index to i64
    %v_158 = llvm.getelementptr %v_21[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_157, %v_158 : i64, !llvm.ptr
    %v_159 = memref.extract_aligned_pointer_as_index %v_153 : memref<?xf64> -> index
    %v_160 = arith.index_cast %v_159 : index to i64
    %v_161 = llvm.inttoptr %v_160 : i64 to !llvm.ptr
    %v_162 = llvm.getelementptr %v_21[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_161, %v_162 : !llvm.ptr, !llvm.ptr
    %v_163 = llvm.load %v_52 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_164 = builtin.unrealized_conversion_cast %v_163 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_165 = llvm.load %v_52 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_166 = builtin.unrealized_conversion_cast %v_165 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_167 = memref.dim %v_166, %v_75 : memref<?xf64>
    %v_168 = arith.index_cast %v_167 : index to i64
    %v_169 = llvm.getelementptr %v_39[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_168, %v_169 : i64, !llvm.ptr
    %v_170 = memref.extract_aligned_pointer_as_index %v_164 : memref<?xf64> -> index
    %v_171 = arith.index_cast %v_170 : index to i64
    %v_172 = llvm.inttoptr %v_171 : i64 to !llvm.ptr
    %v_173 = llvm.getelementptr %v_39[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_172, %v_173 : !llvm.ptr, !llvm.ptr
    %v_174 = llvm.load %v_70 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_175 = builtin.unrealized_conversion_cast %v_174 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_176 = llvm.load %v_70 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_177 = builtin.unrealized_conversion_cast %v_176 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_178 = memref.dim %v_177, %v_75 : memref<?xf64>
    %v_179 = arith.index_cast %v_178 : index to i64
    %v_180 = llvm.getelementptr %v_57[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_179, %v_180 : i64, !llvm.ptr
    %v_181 = memref.extract_aligned_pointer_as_index %v_175 : memref<?xf64> -> index
    %v_182 = arith.index_cast %v_181 : index to i64
    %v_183 = llvm.inttoptr %v_182 : i64 to !llvm.ptr
    %v_184 = llvm.getelementptr %v_57[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_183, %v_184 : !llvm.ptr, !llvm.ptr
    %v_185 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_186 = llvm.insertvalue %_A_7, %v_185[0] : !llvm.struct<(!llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_186, %_ret : !llvm.struct<(!llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}
