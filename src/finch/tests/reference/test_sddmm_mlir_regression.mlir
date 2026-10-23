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
    %v_62 = llvm.getelementptr %v[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_63 = llvm.load %v_62 : !llvm.ptr -> i64
    %v_64 = arith.index_cast %v_63 : i64 to index
    %v_65 = llvm.extractvalue %_A_20[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>
    %v_66 = llvm.getelementptr %v_65[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_67 = llvm.load %v_66 : !llvm.ptr -> !llvm.ptr
    %v_68 = llvm.getelementptr %v_67[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_69 = llvm.load %v_68 : !llvm.ptr -> !llvm.ptr
    %v_70 = llvm.getelementptr %v_69[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr)>
    %v_71 = llvm.load %v_70 : !llvm.ptr -> !llvm.ptr
    %v_72 = llvm.getelementptr %v_71[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_73 = llvm.load %v_72 : !llvm.ptr -> !llvm.ptr
    %v_74 = llvm.getelementptr %v_71[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_75 = llvm.load %v_74 : !llvm.ptr -> i64
    %v_76 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_77 = llvm.insertvalue %v_73, %v_76[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_78 = llvm.insertvalue %v_73, %v_77[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_79 = llvm.insertvalue %v_13, %v_78[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_80 = llvm.insertvalue %v_75, %v_79[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_81 = llvm.insertvalue %v_14, %v_80[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_82 = builtin.unrealized_conversion_cast %v_81 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_83 = builtin.unrealized_conversion_cast %v_82 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_84 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_83, %v_84 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_85 = llvm.getelementptr %v_65[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_86 = llvm.load %v_85 : !llvm.ptr -> !llvm.ptr
    %v_87 = llvm.getelementptr %v_86[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_88 = llvm.load %v_87 : !llvm.ptr -> i64
    %v_89 = arith.index_cast %v_88 : i64 to index
    %v_90 = llvm.extractvalue %_A_21[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>
    %v_91 = llvm.getelementptr %v_90[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_92 = llvm.load %v_91 : !llvm.ptr -> !llvm.ptr
    %v_93 = llvm.getelementptr %v_92[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_94 = llvm.load %v_93 : !llvm.ptr -> !llvm.ptr
    %v_95 = llvm.getelementptr %v_94[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr)>
    %v_96 = llvm.load %v_95 : !llvm.ptr -> !llvm.ptr
    %v_97 = llvm.getelementptr %v_96[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_98 = llvm.load %v_97 : !llvm.ptr -> !llvm.ptr
    %v_99 = llvm.getelementptr %v_96[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_100 = llvm.load %v_99 : !llvm.ptr -> i64
    %v_101 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_102 = llvm.insertvalue %v_98, %v_101[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_103 = llvm.insertvalue %v_98, %v_102[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_104 = llvm.insertvalue %v_13, %v_103[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_105 = llvm.insertvalue %v_100, %v_104[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_106 = llvm.insertvalue %v_14, %v_105[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_107 = builtin.unrealized_conversion_cast %v_106 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_108 = builtin.unrealized_conversion_cast %v_107 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_109 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_108, %v_109 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_110 = llvm.getelementptr %v_90[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_111 = llvm.load %v_110 : !llvm.ptr -> !llvm.ptr
    %v_112 = llvm.getelementptr %v_111[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_113 = llvm.load %v_112 : !llvm.ptr -> i64
    %v_114 = arith.index_cast %v_113 : i64 to index
    %v_115 = llvm.getelementptr %_A_9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_116 = llvm.load %v_115 : !llvm.ptr -> !llvm.ptr
    %v_117 = llvm.getelementptr %v_116[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_118 = llvm.load %v_117 : !llvm.ptr -> !llvm.ptr
    %v_119 = llvm.getelementptr %v_116[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_120 = llvm.load %v_119 : !llvm.ptr -> i64
    %v_121 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_122 = llvm.insertvalue %v_118, %v_121[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_123 = llvm.insertvalue %v_118, %v_122[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_124 = llvm.insertvalue %v_13, %v_123[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_125 = llvm.insertvalue %v_120, %v_124[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_126 = llvm.insertvalue %v_14, %v_125[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_127 = builtin.unrealized_conversion_cast %v_126 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_128 = builtin.unrealized_conversion_cast %v_127 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_129 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_128, %v_129 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_130 = llvm.getelementptr %_A_9[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_131 = llvm.load %v_130 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_132 = llvm.extractvalue %v_131[0] : !llvm.struct<(i64, i64)>
    %v_133 = arith.index_cast %v_132 : i64 to index
    %v_134 = llvm.getelementptr %_A_9[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_135 = llvm.load %v_134 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_136 = llvm.extractvalue %v_135[1] : !llvm.struct<(i64, i64)>
    %v_137 = arith.index_cast %v_136 : i64 to index
    %v_138 = arith.constant 0 : index
    %v_139 = llvm.load %v_129 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_140 = builtin.unrealized_conversion_cast %v_139 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_141 = memref.dim %v_140, %v_138 : memref<?xf64>
    %v_142 = arith.constant 1 : index
    scf.for %v_143 = %v_138 to %v_141 step %v_142 {
      %v_144 = llvm.load %v_129 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_145 = builtin.unrealized_conversion_cast %v_144 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_146 = arith.constant 0.0 : f64
      memref.store %v_146, %v_145[%v_143] : memref<?xf64>
    }
    scf.for %v_147 = %v_138 to %v_64 step %v_142 {
      %v_148 = arith.muli %v_133, %v_147 : index
      scf.for %v_149 = %v_138 to %v_89 step %v_142 {
        %v_150 = arith.muli %v_147, %v_89 : index
        %v_151 = arith.addi %v_150, %v_149 : index
        %v_152 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_153 = builtin.unrealized_conversion_cast %v_152 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_154 = memref.load %v_153[%v_147] : memref<?xindex>
        %v_155 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_156 = builtin.unrealized_conversion_cast %v_155 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_157 = arith.addi %v_142, %v_147 : index
        %v_158 = memref.load %v_156[%v_157] : memref<?xindex>
        %v_159 = arith.cmpi slt, %v_154, %v_158 : index
        %v_167, %v_168 = scf.if %v_159 -> (index, index) {
          %v_160 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_161 = builtin.unrealized_conversion_cast %v_160 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_162 = memref.load %v_161[%v_154] : memref<?xindex>
          %v_163 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_164 = builtin.unrealized_conversion_cast %v_163 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_165 = arith.subi %v_158, %v_142 : index
          %v_166 = memref.load %v_164[%v_165] : memref<?xindex>
          scf.yield %v_162, %v_166 : index, index
        } else {
          scf.yield %v_142, %v_138 : index, index
        }
        %v_169 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_170 = builtin.unrealized_conversion_cast %v_169 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_171 = memref.load %v_170[%v_154] : memref<?xindex>
        %v_172 = arith.cmpi slt, %v_171, %v_138 : index
        %v_177 = scf.if %v_172 -> (index) {
          %v_173 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_174 = builtin.unrealized_conversion_cast %v_173 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_175 = arith.subi %v_158, %v_142 : index
          %v_176 = func.call @scansearch_index(%v_174, %v_138, %v_154, %v_175) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_176 : index
        } else {
          scf.yield %v_154 : index
        }
        %v_229:3 = scf.while (%v_178 = %v_138, %v_179 = %v_177, %v_180 = %v_167) : (index, index, index) -> (index, index, index) {
          %v_181 = arith.addi %v_142, %v_168 : index
          %v_182 = arith.minsi %v_27, %v_181 : index
          %v_183 = arith.cmpi slt, %v_180, %v_182 : index
          scf.condition(%v_183) %v_178, %v_179, %v_180 : index, index, index
        } do {
          ^bb(%v_178: index, %v_179: index, %v_180: index):
          %v_184 = arith.constant 1 : i1
          %v_226, %v_227, %v_228 = scf.if %v_184 -> (index, index, index) {
            %v_185 = arith.addi %v_142, %v_180 : index
            %v_186 = arith.minsi %v_185, %v_180 : index
            scf.for %v_187 = %v_178 to %v_186 step %v_142 {
              %v_188 = arith.muli %v_137, %v_187 : index
              %v_189 = arith.addi %v_148, %v_188 : index
              %v_190 = llvm.load %v_129 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_191 = builtin.unrealized_conversion_cast %v_190 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_192 = llvm.load %v_129 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_193 = builtin.unrealized_conversion_cast %v_192 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_194 = memref.load %v_193[%v_189] : memref<?xf64>
              memref.store %v_194, %v_191[%v_189] : memref<?xf64>
            }
            %v_195 = arith.maxsi %v_178, %v_180 : index
            %v_196 = arith.addi %v_142, %v_180 : index
            scf.for %v_197 = %v_195 to %v_196 step %v_142 {
              %v_198 = arith.muli %v_137, %v_197 : index
              %v_199 = arith.addi %v_148, %v_198 : index
              %v_200 = arith.muli %v_149, %v_114 : index
              %v_201 = arith.addi %v_200, %v_197 : index
              %v_202 = llvm.load %v_129 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_203 = builtin.unrealized_conversion_cast %v_202 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_204 = llvm.load %v_129 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_205 = builtin.unrealized_conversion_cast %v_204 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_206 = memref.load %v_205[%v_199] : memref<?xf64>
              %v_207 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_208 = builtin.unrealized_conversion_cast %v_207 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_209 = memref.load %v_208[%v_179] : memref<?xf64>
              %v_210 = llvm.load %v_84 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_211 = builtin.unrealized_conversion_cast %v_210 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_212 = memref.load %v_211[%v_151] : memref<?xf64>
              %v_213 = arith.mulf %v_209, %v_212 : f64
              %v_214 = llvm.load %v_109 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_215 = builtin.unrealized_conversion_cast %v_214 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_216 = memref.load %v_215[%v_201] : memref<?xf64>
              %v_217 = arith.mulf %v_213, %v_216 : f64
              %v_218 = arith.addf %v_206, %v_217 : f64
              memref.store %v_218, %v_203[%v_199] : memref<?xf64>
            }
            %v_219 = arith.addi %v_142, %v_180 : index
            %v_220 = arith.addi %v_142, %v_179 : index
            %v_221 = arith.cmpi slt, %v_220, %v_158 : index
            %v_225 = scf.if %v_221 -> (index) {
              %v_222 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_223 = builtin.unrealized_conversion_cast %v_222 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
              %v_224 = memref.load %v_223[%v_220] : memref<?xindex>
              scf.yield %v_224 : index
            } else {
              scf.yield %v_27 : index
            }
            scf.yield %v_219, %v_220, %v_225 : index, index, index
          } else {
            scf.yield %v_178, %v_179, %v_180 : index, index, index
          }
          scf.yield %v_226, %v_227, %v_228 : index, index, index
        }
        %v_230 = arith.addi %v_142, %v_168 : index
        %v_231 = arith.maxsi %v_138, %v_230 : index
        scf.for %v_232 = %v_231 to %v_27 step %v_142 {
          %v_233 = arith.muli %v_137, %v_232 : index
          %v_234 = arith.addi %v_148, %v_233 : index
          %v_235 = llvm.load %v_129 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_236 = builtin.unrealized_conversion_cast %v_235 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_237 = llvm.load %v_129 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_238 = builtin.unrealized_conversion_cast %v_237 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_239 = memref.load %v_238[%v_234] : memref<?xf64>
          memref.store %v_239, %v_236[%v_234] : memref<?xf64>
        }
      }
    }
    %v_240 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_241 = builtin.unrealized_conversion_cast %v_240 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_242 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_243 = builtin.unrealized_conversion_cast %v_242 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_244 = memref.dim %v_243, %v_138 : memref<?xf64>
    %v_245 = arith.index_cast %v_244 : index to i64
    %v_246 = llvm.getelementptr %v_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_245, %v_246 : i64, !llvm.ptr
    %v_247 = memref.extract_aligned_pointer_as_index %v_241 : memref<?xf64> -> index
    %v_248 = arith.index_cast %v_247 : index to i64
    %v_249 = llvm.inttoptr %v_248 : i64 to !llvm.ptr
    %v_250 = llvm.getelementptr %v_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_249, %v_250 : !llvm.ptr, !llvm.ptr
    %v_251 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_252 = builtin.unrealized_conversion_cast %v_251 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_253 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_254 = builtin.unrealized_conversion_cast %v_253 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_255 = memref.dim %v_254, %v_138 : memref<?xindex>
    %v_256 = arith.index_cast %v_255 : index to i64
    %v_257 = llvm.getelementptr %v_31[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_256, %v_257 : i64, !llvm.ptr
    %v_258 = memref.extract_aligned_pointer_as_index %v_252 : memref<?xindex> -> index
    %v_259 = arith.index_cast %v_258 : index to i64
    %v_260 = llvm.inttoptr %v_259 : i64 to !llvm.ptr
    %v_261 = llvm.getelementptr %v_31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_260, %v_261 : !llvm.ptr, !llvm.ptr
    %v_262 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_263 = builtin.unrealized_conversion_cast %v_262 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_264 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_265 = builtin.unrealized_conversion_cast %v_264 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_266 = memref.dim %v_265, %v_138 : memref<?xindex>
    %v_267 = arith.index_cast %v_266 : index to i64
    %v_268 = llvm.getelementptr %v_48[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_267, %v_268 : i64, !llvm.ptr
    %v_269 = memref.extract_aligned_pointer_as_index %v_263 : memref<?xindex> -> index
    %v_270 = arith.index_cast %v_269 : index to i64
    %v_271 = llvm.inttoptr %v_270 : i64 to !llvm.ptr
    %v_272 = llvm.getelementptr %v_48[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_271, %v_272 : !llvm.ptr, !llvm.ptr
    %v_273 = llvm.load %v_84 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_274 = builtin.unrealized_conversion_cast %v_273 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_275 = llvm.load %v_84 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_276 = builtin.unrealized_conversion_cast %v_275 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_277 = memref.dim %v_276, %v_138 : memref<?xf64>
    %v_278 = arith.index_cast %v_277 : index to i64
    %v_279 = llvm.getelementptr %v_71[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_278, %v_279 : i64, !llvm.ptr
    %v_280 = memref.extract_aligned_pointer_as_index %v_274 : memref<?xf64> -> index
    %v_281 = arith.index_cast %v_280 : index to i64
    %v_282 = llvm.inttoptr %v_281 : i64 to !llvm.ptr
    %v_283 = llvm.getelementptr %v_71[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_282, %v_283 : !llvm.ptr, !llvm.ptr
    %v_284 = llvm.load %v_109 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_285 = builtin.unrealized_conversion_cast %v_284 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_286 = llvm.load %v_109 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_287 = builtin.unrealized_conversion_cast %v_286 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_288 = memref.dim %v_287, %v_138 : memref<?xf64>
    %v_289 = arith.index_cast %v_288 : index to i64
    %v_290 = llvm.getelementptr %v_96[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_289, %v_290 : i64, !llvm.ptr
    %v_291 = memref.extract_aligned_pointer_as_index %v_285 : memref<?xf64> -> index
    %v_292 = arith.index_cast %v_291 : index to i64
    %v_293 = llvm.inttoptr %v_292 : i64 to !llvm.ptr
    %v_294 = llvm.getelementptr %v_96[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_293, %v_294 : !llvm.ptr, !llvm.ptr
    %v_295 = llvm.load %v_129 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_296 = builtin.unrealized_conversion_cast %v_295 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_297 = llvm.load %v_129 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_298 = builtin.unrealized_conversion_cast %v_297 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_299 = memref.dim %v_298, %v_138 : memref<?xf64>
    %v_300 = arith.index_cast %v_299 : index to i64
    %v_301 = llvm.getelementptr %v_116[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_300, %v_301 : i64, !llvm.ptr
    %v_302 = memref.extract_aligned_pointer_as_index %v_296 : memref<?xf64> -> index
    %v_303 = arith.index_cast %v_302 : index to i64
    %v_304 = llvm.inttoptr %v_303 : i64 to !llvm.ptr
    %v_305 = llvm.getelementptr %v_116[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_304, %v_305 : !llvm.ptr, !llvm.ptr
    %v_306 = llvm.mlir.undef : !llvm.struct<(!llvm.ptr)>
    %v_307 = llvm.insertvalue %_A_9, %v_306[0] : !llvm.struct<(!llvm.ptr)>
    llvm.store %v_307, %_ret : !llvm.struct<(!llvm.ptr)>, !llvm.ptr
    func.return
  }
}
