Compiling MLIR code:
module {
  func.func @scansearch_index(
    %arr: memref<?xindex>, %x: index,
    %lo: index, %hi: index
  ) -> index attributes {llvm.emit_c_interface} {
    %1 = arith.constant 1 : index
    %g:2 = scf.while (%d = %1, %p = %lo) : (index, index) -> (index, index) {
      %plt = arith.cmpi slt, %p, %hi : index
      %cond = scf.if %plt -> (i1) {
        %ap = memref.load %arr[%p] : memref<?xindex>
        %al = arith.cmpi slt, %ap, %x : index
        scf.yield %al : i1
      } else {
        %f = arith.constant false
        scf.yield %f : i1
      }
      scf.condition(%cond) %d, %p : index, index
    } do {
    ^bb0(%d: index, %p: index):
      %d2 = arith.shli %d, %1 : index
      %p2 = arith.addi %p, %d2 : index
      scf.yield %d2, %p2 : index, index
    }
    %lo1 = arith.subi %g#1, %g#0 : index
    %minp = arith.minsi %g#1, %hi : index
    %hi1 = arith.addi %minp, %1 : index
    %b:2 = scf.while (%l = %lo1, %h = %hi1) : (index, index) -> (index, index) {
      %hm1 = arith.subi %h, %1 : index
      %go = arith.cmpi slt, %l, %hm1 : index
      scf.condition(%go) %l, %h : index, index
    } do {
    ^bb0(%l: index, %h: index):
      %diff = arith.subi %h, %l : index
      %half = arith.shrsi %diff, %1 : index
      %m = arith.addi %l, %half : index
      %am = memref.load %arr[%m] : memref<?xindex>
      %al = arith.cmpi slt, %am, %x : index
      %l2, %h2 = scf.if %al -> (index, index) {
        scf.yield %m, %h : index, index
      } else {
        scf.yield %l, %m : index, index
      }
      scf.yield %l2, %h2 : index, index
    }
    return %b#1 : index
  }

  func.func @main(%_A_19: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>, %_A_20: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>, %_A_21: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>, %_A_9: !llvm.ptr, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_19[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>
    %v_2 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_3 = llvm.load %v_2 : !llvm.ptr -> !llvm.ptr
    %v_4 = llvm.getelementptr %v_3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_5 = llvm.load %v_4 : !llvm.ptr -> !llvm.ptr
    %v_6 = llvm.getelementptr %v_5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr)>
    %v_7 = llvm.load %v_6 : !llvm.ptr -> !llvm.ptr
    %v_8 = llvm.getelementptr %v_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_9 = llvm.load %v_8 : !llvm.ptr -> !llvm.ptr
    %v_10 = llvm.getelementptr %v_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_11 = llvm.load %v_10 : !llvm.ptr -> i64
    %v_12 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_13 = arith.constant 0 : i64
    %v_14 = arith.constant 1 : i64
    %v_15 = llvm.insertvalue %v_9, %v_12[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_16 = llvm.insertvalue %v_9, %v_15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_17 = llvm.insertvalue %v_13, %v_16[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_18 = llvm.insertvalue %v_11, %v_17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_19 = llvm.insertvalue %v_14, %v_18[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_20 = builtin.unrealized_conversion_cast %v_19 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_21 = builtin.unrealized_conversion_cast %v_20 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_22 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_21, %v_22 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_23 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_24 = llvm.load %v_23 : !llvm.ptr -> !llvm.ptr
    %v_25 = llvm.getelementptr %v_24[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_26 = llvm.load %v_25 : !llvm.ptr -> i64
    %v_27 = arith.index_cast %v_26 : i64 to index
    %v_28 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_29 = llvm.load %v_28 : !llvm.ptr -> !llvm.ptr
    %v_30 = llvm.getelementptr %v_29[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_31 = llvm.load %v_30 : !llvm.ptr -> !llvm.ptr
    %v_32 = llvm.getelementptr %v_31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_33 = llvm.load %v_32 : !llvm.ptr -> !llvm.ptr
    %v_34 = llvm.getelementptr %v_31[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_35 = llvm.load %v_34 : !llvm.ptr -> i64
    %v_36 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_37 = llvm.insertvalue %v_33, %v_36[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_38 = llvm.insertvalue %v_33, %v_37[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_39 = llvm.insertvalue %v_13, %v_38[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_40 = llvm.insertvalue %v_35, %v_39[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_41 = llvm.insertvalue %v_14, %v_40[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_42 = builtin.unrealized_conversion_cast %v_41 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_43 = builtin.unrealized_conversion_cast %v_42 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_44 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_43, %v_44 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_45 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_46 = llvm.load %v_45 : !llvm.ptr -> !llvm.ptr
    %v_47 = llvm.getelementptr %v_46[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_48 = llvm.load %v_47 : !llvm.ptr -> !llvm.ptr
    %v_49 = llvm.getelementptr %v_48[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_50 = llvm.load %v_49 : !llvm.ptr -> !llvm.ptr
    %v_51 = llvm.getelementptr %v_48[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_52 = llvm.load %v_51 : !llvm.ptr -> i64
    %v_53 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_54 = llvm.insertvalue %v_50, %v_53[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_55 = llvm.insertvalue %v_50, %v_54[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_56 = llvm.insertvalue %v_13, %v_55[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_57 = llvm.insertvalue %v_52, %v_56[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_58 = llvm.insertvalue %v_14, %v_57[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_59 = builtin.unrealized_conversion_cast %v_58 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_60 = builtin.unrealized_conversion_cast %v_59 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_61 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_60, %v_61 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_62 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_63 = llvm.load %v_62 : !llvm.ptr -> !llvm.ptr
    %v_64 = llvm.getelementptr %v_63[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_65 = llvm.load %v_64 : !llvm.ptr -> i64
    %v_66 = arith.index_cast %v_65 : i64 to index
    %v_67 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_68 = llvm.load %v_67 : !llvm.ptr -> !llvm.ptr
    %v_69 = llvm.getelementptr %v_68[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_70 = llvm.load %v_69 : !llvm.ptr -> i64
    %v_71 = arith.index_cast %v_70 : i64 to index
    %v_72 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_73 = llvm.load %v_72 : !llvm.ptr -> !llvm.ptr
    %v_74 = llvm.getelementptr %v_73[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_75 = llvm.load %v_74 : !llvm.ptr -> i64
    %v_76 = arith.index_cast %v_75 : i64 to index
    %v_77 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_78 = llvm.load %v_77 : !llvm.ptr -> !llvm.ptr
    %v_79 = llvm.getelementptr %v_78[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_80 = llvm.load %v_79 : !llvm.ptr -> i1
    %v_81 = llvm.getelementptr %v[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_82 = llvm.load %v_81 : !llvm.ptr -> i64
    %v_83 = arith.index_cast %v_82 : i64 to index
    %v_84 = llvm.getelementptr %v[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_85 = llvm.load %v_84 : !llvm.ptr -> i64
    %v_86 = arith.index_cast %v_85 : i64 to index
    %v_87 = llvm.extractvalue %_A_20[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>
    %v_88 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_89 = llvm.load %v_88 : !llvm.ptr -> !llvm.ptr
    %v_90 = llvm.getelementptr %v_89[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_91 = llvm.load %v_90 : !llvm.ptr -> !llvm.ptr
    %v_92 = llvm.getelementptr %v_91[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr)>
    %v_93 = llvm.load %v_92 : !llvm.ptr -> !llvm.ptr
    %v_94 = llvm.getelementptr %v_93[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_95 = llvm.load %v_94 : !llvm.ptr -> !llvm.ptr
    %v_96 = llvm.getelementptr %v_93[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_97 = llvm.load %v_96 : !llvm.ptr -> i64
    %v_98 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_99 = llvm.insertvalue %v_95, %v_98[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_100 = llvm.insertvalue %v_95, %v_99[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_101 = llvm.insertvalue %v_13, %v_100[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_102 = llvm.insertvalue %v_97, %v_101[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_103 = llvm.insertvalue %v_14, %v_102[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_104 = builtin.unrealized_conversion_cast %v_103 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_105 = builtin.unrealized_conversion_cast %v_104 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_106 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_105, %v_106 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_107 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_108 = llvm.load %v_107 : !llvm.ptr -> !llvm.ptr
    %v_109 = llvm.getelementptr %v_108[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_110 = llvm.load %v_109 : !llvm.ptr -> i64
    %v_111 = arith.index_cast %v_110 : i64 to index
    %v_112 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_113 = llvm.load %v_112 : !llvm.ptr -> !llvm.ptr
    %v_114 = llvm.getelementptr %v_113[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_115 = llvm.load %v_114 : !llvm.ptr -> i64
    %v_116 = arith.index_cast %v_115 : i64 to index
    %v_117 = llvm.getelementptr %v_87[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_118 = llvm.load %v_117 : !llvm.ptr -> i64
    %v_119 = arith.index_cast %v_118 : i64 to index
    %v_120 = llvm.getelementptr %v_87[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_121 = llvm.load %v_120 : !llvm.ptr -> i64
    %v_122 = arith.index_cast %v_121 : i64 to index
    %v_123 = llvm.extractvalue %_A_21[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>
    %v_124 = llvm.getelementptr %v_123[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_125 = llvm.load %v_124 : !llvm.ptr -> !llvm.ptr
    %v_126 = llvm.getelementptr %v_125[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_127 = llvm.load %v_126 : !llvm.ptr -> !llvm.ptr
    %v_128 = llvm.getelementptr %v_127[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr)>
    %v_129 = llvm.load %v_128 : !llvm.ptr -> !llvm.ptr
    %v_130 = llvm.getelementptr %v_129[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_131 = llvm.load %v_130 : !llvm.ptr -> !llvm.ptr
    %v_132 = llvm.getelementptr %v_129[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_133 = llvm.load %v_132 : !llvm.ptr -> i64
    %v_134 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_135 = llvm.insertvalue %v_131, %v_134[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_136 = llvm.insertvalue %v_131, %v_135[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_137 = llvm.insertvalue %v_13, %v_136[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_138 = llvm.insertvalue %v_133, %v_137[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_139 = llvm.insertvalue %v_14, %v_138[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_140 = builtin.unrealized_conversion_cast %v_139 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_141 = builtin.unrealized_conversion_cast %v_140 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_142 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_141, %v_142 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_143 = llvm.getelementptr %v_123[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_144 = llvm.load %v_143 : !llvm.ptr -> !llvm.ptr
    %v_145 = llvm.getelementptr %v_144[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_146 = llvm.load %v_145 : !llvm.ptr -> i64
    %v_147 = arith.index_cast %v_146 : i64 to index
    %v_148 = llvm.getelementptr %v_123[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_149 = llvm.load %v_148 : !llvm.ptr -> !llvm.ptr
    %v_150 = llvm.getelementptr %v_149[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_151 = llvm.load %v_150 : !llvm.ptr -> i64
    %v_152 = arith.index_cast %v_151 : i64 to index
    %v_153 = llvm.getelementptr %v_123[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_154 = llvm.load %v_153 : !llvm.ptr -> i64
    %v_155 = arith.index_cast %v_154 : i64 to index
    %v_156 = llvm.getelementptr %v_123[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_157 = llvm.load %v_156 : !llvm.ptr -> i64
    %v_158 = arith.index_cast %v_157 : i64 to index
    %v_159 = llvm.getelementptr %_A_9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_160 = llvm.load %v_159 : !llvm.ptr -> !llvm.ptr
    %v_161 = llvm.getelementptr %v_160[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_162 = llvm.load %v_161 : !llvm.ptr -> !llvm.ptr
    %v_163 = llvm.getelementptr %v_160[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_164 = llvm.load %v_163 : !llvm.ptr -> i64
    %v_165 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_166 = llvm.insertvalue %v_162, %v_165[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_167 = llvm.insertvalue %v_162, %v_166[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_168 = llvm.insertvalue %v_13, %v_167[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_169 = llvm.insertvalue %v_164, %v_168[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_170 = llvm.insertvalue %v_14, %v_169[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_171 = builtin.unrealized_conversion_cast %v_170 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_172 = builtin.unrealized_conversion_cast %v_171 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_173 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_172, %v_173 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_174 = llvm.getelementptr %_A_9[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_175 = llvm.load %v_174 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_176 = llvm.extractvalue %v_175[0] : !llvm.struct<(i64, i64)>
    %v_177 = arith.index_cast %v_176 : i64 to index
    %v_178 = llvm.getelementptr %_A_9[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_179 = llvm.load %v_178 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_180 = llvm.extractvalue %v_179[1] : !llvm.struct<(i64, i64)>
    %v_181 = arith.index_cast %v_180 : i64 to index
    %v_182 = llvm.getelementptr %_A_9[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_183 = llvm.load %v_182 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_184 = llvm.extractvalue %v_183[0] : !llvm.struct<(i64, i64)>
    %v_185 = arith.index_cast %v_184 : i64 to index
    %v_186 = llvm.getelementptr %_A_9[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_187 = llvm.load %v_186 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_188 = llvm.extractvalue %v_187[1] : !llvm.struct<(i64, i64)>
    %v_189 = arith.index_cast %v_188 : i64 to index
    %v_190 = arith.constant 0 : index
    %v_191 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_192 = builtin.unrealized_conversion_cast %v_191 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_193 = memref.dim %v_192, %v_190 : memref<?xf64>
    %v_194 = arith.constant 1 : index
    scf.for %v_195 = %v_190 to %v_193 step %v_194 {
      %v_196 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_197 = builtin.unrealized_conversion_cast %v_196 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_198 = arith.constant 0.0 : f64
      memref.store %v_198, %v_197[%v_195] : memref<?xf64>
    }
    scf.for %v_199 = %v_190 to %v_83 step %v_194 {
      %v_200 = arith.muli %v_185, %v_199 : index
      scf.for %v_201 = %v_190 to %v_111 step %v_194 {
        %v_202 = arith.muli %v_199, %v_111 : index
        %v_203 = arith.addi %v_202, %v_201 : index
        %v_204 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_205 = builtin.unrealized_conversion_cast %v_204 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_206 = memref.load %v_205[%v_199] : memref<?xindex>
        %v_207 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_208 = builtin.unrealized_conversion_cast %v_207 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_209 = arith.addi %v_194, %v_199 : index
        %v_210 = memref.load %v_208[%v_209] : memref<?xindex>
        %v_211 = arith.cmpi slt, %v_206, %v_210 : index
        %v_219, %v_220 = scf.if %v_211 -> (index, index) {
          %v_212 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_213 = builtin.unrealized_conversion_cast %v_212 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_214 = memref.load %v_213[%v_206] : memref<?xindex>
          %v_215 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_216 = builtin.unrealized_conversion_cast %v_215 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_217 = arith.subi %v_210, %v_194 : index
          %v_218 = memref.load %v_216[%v_217] : memref<?xindex>
          scf.yield %v_214, %v_218 : index, index
        } else {
          scf.yield %v_194, %v_190 : index, index
        }
        %v_221 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_222 = builtin.unrealized_conversion_cast %v_221 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_223 = memref.load %v_222[%v_206] : memref<?xindex>
        %v_224 = arith.cmpi slt, %v_223, %v_190 : index
        %v_229 = scf.if %v_224 -> (index) {
          %v_225 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_226 = builtin.unrealized_conversion_cast %v_225 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_227 = arith.subi %v_210, %v_194 : index
          %v_228 = func.call @scansearch_index(%v_226, %v_190, %v_206, %v_227) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_228 : index
        } else {
          scf.yield %v_206 : index
        }
        %v_277:3 = scf.while (%v_230 = %v_190, %v_231 = %v_229, %v_232 = %v_219) : (index, index, index) -> (index, index, index) {
          %v_233 = arith.addi %v_194, %v_220 : index
          %v_234 = arith.minsi %v_27, %v_233 : index
          %v_235 = arith.cmpi slt, %v_232, %v_234 : index
          scf.condition(%v_235) %v_230, %v_231, %v_232 : index, index, index
        } do {
          ^bb(%v_230: index, %v_231: index, %v_232: index):
          %v_236 = arith.addi %v_194, %v_232 : index
          %v_237 = arith.minsi %v_236, %v_232 : index
          scf.for %v_238 = %v_230 to %v_237 step %v_194 {
            %v_239 = arith.muli %v_189, %v_238 : index
            %v_240 = arith.addi %v_200, %v_239 : index
            %v_241 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_242 = builtin.unrealized_conversion_cast %v_241 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_243 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_244 = builtin.unrealized_conversion_cast %v_243 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_245 = memref.load %v_244[%v_240] : memref<?xf64>
            memref.store %v_245, %v_242[%v_240] : memref<?xf64>
          }
          %v_246 = arith.maxsi %v_230, %v_232 : index
          %v_247 = arith.addi %v_194, %v_232 : index
          scf.for %v_248 = %v_246 to %v_247 step %v_194 {
            %v_249 = arith.muli %v_189, %v_248 : index
            %v_250 = arith.addi %v_200, %v_249 : index
            %v_251 = arith.muli %v_201, %v_147 : index
            %v_252 = arith.addi %v_251, %v_248 : index
            %v_253 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_254 = builtin.unrealized_conversion_cast %v_253 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_255 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_256 = builtin.unrealized_conversion_cast %v_255 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_257 = memref.load %v_256[%v_250] : memref<?xf64>
            %v_258 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_259 = builtin.unrealized_conversion_cast %v_258 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_260 = memref.load %v_259[%v_231] : memref<?xf64>
            %v_261 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_262 = builtin.unrealized_conversion_cast %v_261 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_263 = memref.load %v_262[%v_203] : memref<?xf64>
            %v_264 = arith.mulf %v_260, %v_263 : f64
            %v_265 = llvm.load %v_142 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_266 = builtin.unrealized_conversion_cast %v_265 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_267 = memref.load %v_266[%v_252] : memref<?xf64>
            %v_268 = arith.mulf %v_264, %v_267 : f64
            %v_269 = arith.addf %v_257, %v_268 : f64
            memref.store %v_269, %v_254[%v_250] : memref<?xf64>
          }
          %v_270 = arith.addi %v_194, %v_232 : index
          %v_271 = arith.addi %v_194, %v_231 : index
          %v_272 = arith.cmpi slt, %v_271, %v_210 : index
          %v_276 = scf.if %v_272 -> (index) {
            %v_273 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_274 = builtin.unrealized_conversion_cast %v_273 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
            %v_275 = memref.load %v_274[%v_271] : memref<?xindex>
            scf.yield %v_275 : index
          } else {
            scf.yield %v_27 : index
          }
          scf.yield %v_270, %v_271, %v_276 : index, index, index
        }
        %v_278 = arith.addi %v_194, %v_220 : index
        %v_279 = arith.maxsi %v_190, %v_278 : index
        scf.for %v_280 = %v_279 to %v_27 step %v_194 {
          %v_281 = arith.muli %v_189, %v_280 : index
          %v_282 = arith.addi %v_200, %v_281 : index
          %v_283 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_284 = builtin.unrealized_conversion_cast %v_283 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_285 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_286 = builtin.unrealized_conversion_cast %v_285 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_287 = memref.load %v_286[%v_282] : memref<?xf64>
          memref.store %v_287, %v_284[%v_282] : memref<?xf64>
        }
      }
    }
    %v_288 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_289 = builtin.unrealized_conversion_cast %v_288 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_290 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_291 = builtin.unrealized_conversion_cast %v_290 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_292 = memref.dim %v_291, %v_190 : memref<?xf64>
    %v_293 = arith.index_cast %v_292 : index to i64
    %v_294 = llvm.getelementptr %v_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_293, %v_294 : i64, !llvm.ptr
    %v_295 = memref.extract_aligned_pointer_as_index %v_289 : memref<?xf64> -> index
    %v_296 = arith.index_cast %v_295 : index to i64
    %v_297 = llvm.inttoptr %v_296 : i64 to !llvm.ptr
    %v_298 = llvm.getelementptr %v_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_297, %v_298 : !llvm.ptr, !llvm.ptr
    %v_299 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_300 = builtin.unrealized_conversion_cast %v_299 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_301 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_302 = builtin.unrealized_conversion_cast %v_301 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_303 = memref.dim %v_302, %v_190 : memref<?xindex>
    %v_304 = arith.index_cast %v_303 : index to i64
    %v_305 = llvm.getelementptr %v_31[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_304, %v_305 : i64, !llvm.ptr
    %v_306 = memref.extract_aligned_pointer_as_index %v_300 : memref<?xindex> -> index
    %v_307 = arith.index_cast %v_306 : index to i64
    %v_308 = llvm.inttoptr %v_307 : i64 to !llvm.ptr
    %v_309 = llvm.getelementptr %v_31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_308, %v_309 : !llvm.ptr, !llvm.ptr
    %v_310 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_311 = builtin.unrealized_conversion_cast %v_310 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_312 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_313 = builtin.unrealized_conversion_cast %v_312 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_314 = memref.dim %v_313, %v_190 : memref<?xindex>
    %v_315 = arith.index_cast %v_314 : index to i64
    %v_316 = llvm.getelementptr %v_48[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_315, %v_316 : i64, !llvm.ptr
    %v_317 = memref.extract_aligned_pointer_as_index %v_311 : memref<?xindex> -> index
    %v_318 = arith.index_cast %v_317 : index to i64
    %v_319 = llvm.inttoptr %v_318 : i64 to !llvm.ptr
    %v_320 = llvm.getelementptr %v_48[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_319, %v_320 : !llvm.ptr, !llvm.ptr
    %v_321 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_322 = builtin.unrealized_conversion_cast %v_321 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_323 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_324 = builtin.unrealized_conversion_cast %v_323 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_325 = memref.dim %v_324, %v_190 : memref<?xf64>
    %v_326 = arith.index_cast %v_325 : index to i64
    %v_327 = llvm.getelementptr %v_93[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_326, %v_327 : i64, !llvm.ptr
    %v_328 = memref.extract_aligned_pointer_as_index %v_322 : memref<?xf64> -> index
    %v_329 = arith.index_cast %v_328 : index to i64
    %v_330 = llvm.inttoptr %v_329 : i64 to !llvm.ptr
    %v_331 = llvm.getelementptr %v_93[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_330, %v_331 : !llvm.ptr, !llvm.ptr
    %v_332 = llvm.load %v_142 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_333 = builtin.unrealized_conversion_cast %v_332 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_334 = llvm.load %v_142 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_335 = builtin.unrealized_conversion_cast %v_334 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_336 = memref.dim %v_335, %v_190 : memref<?xf64>
    %v_337 = arith.index_cast %v_336 : index to i64
    %v_338 = llvm.getelementptr %v_129[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_337, %v_338 : i64, !llvm.ptr
    %v_339 = memref.extract_aligned_pointer_as_index %v_333 : memref<?xf64> -> index
    %v_340 = arith.index_cast %v_339 : index to i64
    %v_341 = llvm.inttoptr %v_340 : i64 to !llvm.ptr
    %v_342 = llvm.getelementptr %v_129[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_341, %v_342 : !llvm.ptr, !llvm.ptr
    %v_343 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_344 = builtin.unrealized_conversion_cast %v_343 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_345 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_346 = builtin.unrealized_conversion_cast %v_345 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_347 = memref.dim %v_346, %v_190 : memref<?xf64>
    %v_348 = arith.index_cast %v_347 : index to i64
    %v_349 = llvm.getelementptr %v_160[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_348, %v_349 : i64, !llvm.ptr
    %v_350 = memref.extract_aligned_pointer_as_index %v_344 : memref<?xf64> -> index
    %v_351 = arith.index_cast %v_350 : index to i64
    %v_352 = llvm.inttoptr %v_351 : i64 to !llvm.ptr
    %v_353 = llvm.getelementptr %v_160[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_352, %v_353 : !llvm.ptr, !llvm.ptr
    %v_354 = llvm.mlir.undef : !llvm.struct<(!llvm.ptr)>
    %v_355 = llvm.insertvalue %_A_9, %v_354[0] : !llvm.struct<(!llvm.ptr)>
    llvm.store %v_355, %_ret : !llvm.struct<(!llvm.ptr)>, !llvm.ptr
    func.return
  }
}
