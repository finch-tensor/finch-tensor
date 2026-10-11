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

  func.func @main(%_A_15: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>, %_A_16: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>, %_A_18: !llvm.ptr, %_A_7: !llvm.ptr, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_15[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>
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
    %v_87 = llvm.extractvalue %_A_16[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>
    %v_88 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_89 = llvm.load %v_88 : !llvm.ptr -> !llvm.ptr
    %v_90 = llvm.getelementptr %v_89[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
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
    %v_109 = llvm.getelementptr %v_108[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_110 = llvm.load %v_109 : !llvm.ptr -> i64
    %v_111 = arith.index_cast %v_110 : i64 to index
    %v_112 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_113 = llvm.load %v_112 : !llvm.ptr -> !llvm.ptr
    %v_114 = llvm.getelementptr %v_113[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_115 = llvm.load %v_114 : !llvm.ptr -> !llvm.ptr
    %v_116 = llvm.getelementptr %v_115[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_117 = llvm.load %v_116 : !llvm.ptr -> !llvm.ptr
    %v_118 = llvm.getelementptr %v_115[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_119 = llvm.load %v_118 : !llvm.ptr -> i64
    %v_120 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_121 = llvm.insertvalue %v_117, %v_120[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_122 = llvm.insertvalue %v_117, %v_121[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_123 = llvm.insertvalue %v_13, %v_122[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_124 = llvm.insertvalue %v_119, %v_123[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_125 = llvm.insertvalue %v_14, %v_124[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_126 = builtin.unrealized_conversion_cast %v_125 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_127 = builtin.unrealized_conversion_cast %v_126 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_128 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_127, %v_128 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_129 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_130 = llvm.load %v_129 : !llvm.ptr -> !llvm.ptr
    %v_131 = llvm.getelementptr %v_130[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_132 = llvm.load %v_131 : !llvm.ptr -> !llvm.ptr
    %v_133 = llvm.getelementptr %v_132[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_134 = llvm.load %v_133 : !llvm.ptr -> !llvm.ptr
    %v_135 = llvm.getelementptr %v_132[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_136 = llvm.load %v_135 : !llvm.ptr -> i64
    %v_137 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_138 = llvm.insertvalue %v_134, %v_137[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_139 = llvm.insertvalue %v_134, %v_138[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_140 = llvm.insertvalue %v_13, %v_139[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_141 = llvm.insertvalue %v_136, %v_140[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_142 = llvm.insertvalue %v_14, %v_141[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_143 = builtin.unrealized_conversion_cast %v_142 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_144 = builtin.unrealized_conversion_cast %v_143 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_145 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_144, %v_145 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_146 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_147 = llvm.load %v_146 : !llvm.ptr -> !llvm.ptr
    %v_148 = llvm.getelementptr %v_147[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_149 = llvm.load %v_148 : !llvm.ptr -> i64
    %v_150 = arith.index_cast %v_149 : i64 to index
    %v_151 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_152 = llvm.load %v_151 : !llvm.ptr -> !llvm.ptr
    %v_153 = llvm.getelementptr %v_152[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_154 = llvm.load %v_153 : !llvm.ptr -> i64
    %v_155 = arith.index_cast %v_154 : i64 to index
    %v_156 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_157 = llvm.load %v_156 : !llvm.ptr -> !llvm.ptr
    %v_158 = llvm.getelementptr %v_157[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_159 = llvm.load %v_158 : !llvm.ptr -> i64
    %v_160 = arith.index_cast %v_159 : i64 to index
    %v_161 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_162 = llvm.load %v_161 : !llvm.ptr -> !llvm.ptr
    %v_163 = llvm.getelementptr %v_162[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_164 = llvm.load %v_163 : !llvm.ptr -> i1
    %v_165 = llvm.getelementptr %v_87[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_166 = llvm.load %v_165 : !llvm.ptr -> i64
    %v_167 = arith.index_cast %v_166 : i64 to index
    %v_168 = llvm.getelementptr %v_87[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_169 = llvm.load %v_168 : !llvm.ptr -> i64
    %v_170 = arith.index_cast %v_169 : i64 to index
    %v_171 = llvm.getelementptr %_A_18[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_172 = llvm.load %v_171 : !llvm.ptr -> !llvm.ptr
    %v_173 = llvm.getelementptr %v_172[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_174 = llvm.load %v_173 : !llvm.ptr -> !llvm.ptr
    %v_175 = llvm.getelementptr %v_172[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_176 = llvm.load %v_175 : !llvm.ptr -> i64
    %v_177 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_178 = llvm.insertvalue %v_174, %v_177[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_179 = llvm.insertvalue %v_174, %v_178[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_180 = llvm.insertvalue %v_13, %v_179[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_181 = llvm.insertvalue %v_176, %v_180[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_182 = llvm.insertvalue %v_14, %v_181[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_183 = builtin.unrealized_conversion_cast %v_182 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_184 = builtin.unrealized_conversion_cast %v_183 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_185 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_184, %v_185 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_186 = llvm.getelementptr %_A_18[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_187 = llvm.load %v_186 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_188 = llvm.extractvalue %v_187[0] : !llvm.struct<(i64, i64)>
    %v_189 = arith.index_cast %v_188 : i64 to index
    %v_190 = llvm.getelementptr %_A_18[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_191 = llvm.load %v_190 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_192 = llvm.extractvalue %v_191[1] : !llvm.struct<(i64, i64)>
    %v_193 = arith.index_cast %v_192 : i64 to index
    %v_194 = llvm.getelementptr %_A_18[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_195 = llvm.load %v_194 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_196 = llvm.extractvalue %v_195[0] : !llvm.struct<(i64, i64)>
    %v_197 = arith.index_cast %v_196 : i64 to index
    %v_198 = llvm.getelementptr %_A_18[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_199 = llvm.load %v_198 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_200 = llvm.extractvalue %v_199[1] : !llvm.struct<(i64, i64)>
    %v_201 = arith.index_cast %v_200 : i64 to index
    %v_202 = llvm.getelementptr %_A_7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_203 = llvm.load %v_202 : !llvm.ptr -> !llvm.ptr
    %v_204 = llvm.getelementptr %v_203[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_205 = llvm.load %v_204 : !llvm.ptr -> !llvm.ptr
    %v_206 = llvm.getelementptr %v_203[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_207 = llvm.load %v_206 : !llvm.ptr -> i64
    %v_208 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_209 = llvm.insertvalue %v_205, %v_208[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_210 = llvm.insertvalue %v_205, %v_209[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_211 = llvm.insertvalue %v_13, %v_210[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_212 = llvm.insertvalue %v_207, %v_211[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_213 = llvm.insertvalue %v_14, %v_212[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_214 = builtin.unrealized_conversion_cast %v_213 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_215 = builtin.unrealized_conversion_cast %v_214 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_216 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_215, %v_216 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_217 = llvm.getelementptr %_A_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_218 = llvm.load %v_217 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_219 = llvm.extractvalue %v_218[0] : !llvm.struct<(i64, i64)>
    %v_220 = arith.index_cast %v_219 : i64 to index
    %v_221 = llvm.getelementptr %_A_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_222 = llvm.load %v_221 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_223 = llvm.extractvalue %v_222[1] : !llvm.struct<(i64, i64)>
    %v_224 = arith.index_cast %v_223 : i64 to index
    %v_225 = llvm.getelementptr %_A_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_226 = llvm.load %v_225 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_227 = llvm.extractvalue %v_226[0] : !llvm.struct<(i64, i64)>
    %v_228 = arith.index_cast %v_227 : i64 to index
    %v_229 = llvm.getelementptr %_A_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_230 = llvm.load %v_229 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_231 = llvm.extractvalue %v_230[1] : !llvm.struct<(i64, i64)>
    %v_232 = arith.index_cast %v_231 : i64 to index
    %v_233 = arith.constant 0 : index
    %v_234 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_235 = builtin.unrealized_conversion_cast %v_234 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_236 = memref.dim %v_235, %v_233 : memref<?xf64>
    %v_237 = arith.constant 1 : index
    scf.for %v_238 = %v_233 to %v_236 step %v_237 {
      %v_239 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_240 = builtin.unrealized_conversion_cast %v_239 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_241 = arith.constant 0.0 : f64
      memref.store %v_241, %v_240[%v_238] : memref<?xf64>
    }
    scf.for %v_242 = %v_233 to %v_27 step %v_237 {
      %v_243 = arith.muli %v_197, %v_242 : index
      scf.for %v_244 = %v_233 to %v_83 step %v_237 {
        %v_245 = arith.muli %v_201, %v_244 : index
        %v_246 = arith.addi %v_243, %v_245 : index
        %v_247 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_248 = builtin.unrealized_conversion_cast %v_247 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_249 = memref.load %v_248[%v_244] : memref<?xindex>
        %v_250 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_251 = builtin.unrealized_conversion_cast %v_250 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_252 = arith.addi %v_237, %v_244 : index
        %v_253 = memref.load %v_251[%v_252] : memref<?xindex>
        %v_254 = arith.cmpi slt, %v_249, %v_253 : index
        %v_262, %v_263 = scf.if %v_254 -> (index, index) {
          %v_255 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_256 = builtin.unrealized_conversion_cast %v_255 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_257 = memref.load %v_256[%v_249] : memref<?xindex>
          %v_258 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_259 = builtin.unrealized_conversion_cast %v_258 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_260 = arith.subi %v_253, %v_237 : index
          %v_261 = memref.load %v_259[%v_260] : memref<?xindex>
          scf.yield %v_257, %v_261 : index, index
        } else {
          scf.yield %v_237, %v_233 : index, index
        }
        %v_264 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_265 = builtin.unrealized_conversion_cast %v_264 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_266 = memref.load %v_265[%v_249] : memref<?xindex>
        %v_267 = arith.cmpi slt, %v_266, %v_233 : index
        %v_272 = scf.if %v_267 -> (index) {
          %v_268 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_269 = builtin.unrealized_conversion_cast %v_268 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_270 = arith.subi %v_253, %v_237 : index
          %v_271 = func.call @scansearch_index(%v_269, %v_233, %v_249, %v_270) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_271 : index
        } else {
          scf.yield %v_249 : index
        }
        %v_299:3 = scf.while (%v_273 = %v_233, %v_274 = %v_272, %v_275 = %v_262) : (index, index, index) -> (index, index, index) {
          %v_276 = arith.addi %v_237, %v_263 : index
          %v_277 = arith.minsi %v_27, %v_276 : index
          %v_278 = arith.cmpi slt, %v_275, %v_277 : index
          scf.condition(%v_278) %v_273, %v_274, %v_275 : index, index, index
        } do {
          ^bb(%v_273: index, %v_274: index, %v_275: index):
          %v_279 = arith.constant 1 : i1
          %v_296, %v_297, %v_298 = scf.if %v_279 -> (index, index, index) {
            %v_280 = arith.maxsi %v_273, %v_275 : index
            %v_281 = arith.addi %v_237, %v_275 : index
            scf.for %v_282 = %v_280 to %v_281 step %v_237 {
              %v_283 = arith.cmpi eq, %v_282, %v_242 : index
              scf.if %v_283 {
                %v_284 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
                %v_285 = builtin.unrealized_conversion_cast %v_284 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
                %v_286 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
                %v_287 = builtin.unrealized_conversion_cast %v_286 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
                %v_288 = memref.load %v_287[%v_274] : memref<?xf64>
                memref.store %v_288, %v_285[%v_246] : memref<?xf64>
              }
            }
            %v_289 = arith.addi %v_237, %v_275 : index
            %v_290 = arith.addi %v_237, %v_274 : index
            %v_291 = arith.cmpi slt, %v_290, %v_253 : index
            %v_295 = scf.if %v_291 -> (index) {
              %v_292 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_293 = builtin.unrealized_conversion_cast %v_292 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
              %v_294 = memref.load %v_293[%v_290] : memref<?xindex>
              scf.yield %v_294 : index
            } else {
              scf.yield %v_27 : index
            }
            scf.yield %v_289, %v_290, %v_295 : index, index, index
          } else {
            scf.yield %v_273, %v_274, %v_275 : index, index, index
          }
          scf.yield %v_296, %v_297, %v_298 : index, index, index
        }
      }
    }
    %v_300 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_301 = builtin.unrealized_conversion_cast %v_300 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_302 = memref.dim %v_301, %v_233 : memref<?xf64>
    scf.for %v_303 = %v_233 to %v_302 step %v_237 {
      %v_304 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_305 = builtin.unrealized_conversion_cast %v_304 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_306 = arith.constant 0.0 : f64
      memref.store %v_306, %v_305[%v_303] : memref<?xf64>
    }
    scf.for %v_307 = %v_233 to %v_167 step %v_237 {
      %v_308 = arith.muli %v_197, %v_307 : index
      scf.for %v_309 = %v_233 to %v_193 step %v_237 {
        %v_310 = arith.muli %v_228, %v_309 : index
        %v_311 = arith.muli %v_201, %v_309 : index
        %v_312 = arith.addi %v_308, %v_311 : index
        %v_313 = llvm.load %v_128 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_314 = builtin.unrealized_conversion_cast %v_313 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_315 = memref.load %v_314[%v_307] : memref<?xindex>
        %v_316 = llvm.load %v_128 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_317 = builtin.unrealized_conversion_cast %v_316 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_318 = arith.addi %v_237, %v_307 : index
        %v_319 = memref.load %v_317[%v_318] : memref<?xindex>
        %v_320 = arith.cmpi slt, %v_315, %v_319 : index
        %v_328, %v_329 = scf.if %v_320 -> (index, index) {
          %v_321 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_322 = builtin.unrealized_conversion_cast %v_321 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_323 = memref.load %v_322[%v_315] : memref<?xindex>
          %v_324 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_325 = builtin.unrealized_conversion_cast %v_324 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_326 = arith.subi %v_319, %v_237 : index
          %v_327 = memref.load %v_325[%v_326] : memref<?xindex>
          scf.yield %v_323, %v_327 : index, index
        } else {
          scf.yield %v_237, %v_233 : index, index
        }
        %v_330 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_331 = builtin.unrealized_conversion_cast %v_330 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_332 = memref.load %v_331[%v_315] : memref<?xindex>
        %v_333 = arith.cmpi slt, %v_332, %v_233 : index
        %v_338 = scf.if %v_333 -> (index) {
          %v_334 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_335 = builtin.unrealized_conversion_cast %v_334 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_336 = arith.subi %v_319, %v_237 : index
          %v_337 = func.call @scansearch_index(%v_335, %v_233, %v_315, %v_336) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_337 : index
        } else {
          scf.yield %v_315 : index
        }
        %v_384:3 = scf.while (%v_339 = %v_233, %v_340 = %v_338, %v_341 = %v_328) : (index, index, index) -> (index, index, index) {
          %v_342 = arith.addi %v_237, %v_329 : index
          %v_343 = arith.minsi %v_111, %v_342 : index
          %v_344 = arith.cmpi slt, %v_341, %v_343 : index
          scf.condition(%v_344) %v_339, %v_340, %v_341 : index, index, index
        } do {
          ^bb_2(%v_339: index, %v_340: index, %v_341: index):
          %v_345 = arith.constant 1 : i1
          %v_381, %v_382, %v_383 = scf.if %v_345 -> (index, index, index) {
            %v_346 = arith.addi %v_237, %v_341 : index
            %v_347 = arith.minsi %v_346, %v_341 : index
            scf.for %v_348 = %v_339 to %v_347 step %v_237 {
              %v_349 = arith.muli %v_232, %v_348 : index
              %v_350 = arith.addi %v_310, %v_349 : index
              %v_351 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_352 = builtin.unrealized_conversion_cast %v_351 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_353 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_354 = builtin.unrealized_conversion_cast %v_353 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_355 = memref.load %v_354[%v_350] : memref<?xf64>
              memref.store %v_355, %v_352[%v_350] : memref<?xf64>
            }
            %v_356 = arith.maxsi %v_339, %v_341 : index
            %v_357 = arith.addi %v_237, %v_341 : index
            scf.for %v_358 = %v_356 to %v_357 step %v_237 {
              %v_359 = arith.muli %v_232, %v_358 : index
              %v_360 = arith.addi %v_310, %v_359 : index
              %v_361 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_362 = builtin.unrealized_conversion_cast %v_361 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_363 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_364 = builtin.unrealized_conversion_cast %v_363 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_365 = memref.load %v_364[%v_360] : memref<?xf64>
              %v_366 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_367 = builtin.unrealized_conversion_cast %v_366 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_368 = memref.load %v_367[%v_312] : memref<?xf64>
              %v_369 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_370 = builtin.unrealized_conversion_cast %v_369 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_371 = memref.load %v_370[%v_340] : memref<?xf64>
              %v_372 = arith.mulf %v_368, %v_371 : f64
              %v_373 = arith.addf %v_365, %v_372 : f64
              memref.store %v_373, %v_362[%v_360] : memref<?xf64>
            }
            %v_374 = arith.addi %v_237, %v_341 : index
            %v_375 = arith.addi %v_237, %v_340 : index
            %v_376 = arith.cmpi slt, %v_375, %v_319 : index
            %v_380 = scf.if %v_376 -> (index) {
              %v_377 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_378 = builtin.unrealized_conversion_cast %v_377 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
              %v_379 = memref.load %v_378[%v_375] : memref<?xindex>
              scf.yield %v_379 : index
            } else {
              scf.yield %v_111 : index
            }
            scf.yield %v_374, %v_375, %v_380 : index, index, index
          } else {
            scf.yield %v_339, %v_340, %v_341 : index, index, index
          }
          scf.yield %v_381, %v_382, %v_383 : index, index, index
        }
        %v_385 = arith.addi %v_237, %v_329 : index
        %v_386 = arith.maxsi %v_233, %v_385 : index
        scf.for %v_387 = %v_386 to %v_111 step %v_237 {
          %v_388 = arith.muli %v_232, %v_387 : index
          %v_389 = arith.addi %v_310, %v_388 : index
          %v_390 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_391 = builtin.unrealized_conversion_cast %v_390 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_392 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_393 = builtin.unrealized_conversion_cast %v_392 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_394 = memref.load %v_393[%v_389] : memref<?xf64>
          memref.store %v_394, %v_391[%v_389] : memref<?xf64>
        }
      }
    }
    %v_395 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_396 = builtin.unrealized_conversion_cast %v_395 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_397 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_398 = builtin.unrealized_conversion_cast %v_397 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_399 = memref.dim %v_398, %v_233 : memref<?xf64>
    %v_400 = arith.index_cast %v_399 : index to i64
    %v_401 = llvm.getelementptr %v_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_400, %v_401 : i64, !llvm.ptr
    %v_402 = memref.extract_aligned_pointer_as_index %v_396 : memref<?xf64> -> index
    %v_403 = arith.index_cast %v_402 : index to i64
    %v_404 = llvm.inttoptr %v_403 : i64 to !llvm.ptr
    %v_405 = llvm.getelementptr %v_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_404, %v_405 : !llvm.ptr, !llvm.ptr
    %v_406 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_407 = builtin.unrealized_conversion_cast %v_406 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_408 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_409 = builtin.unrealized_conversion_cast %v_408 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_410 = memref.dim %v_409, %v_233 : memref<?xindex>
    %v_411 = arith.index_cast %v_410 : index to i64
    %v_412 = llvm.getelementptr %v_31[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_411, %v_412 : i64, !llvm.ptr
    %v_413 = memref.extract_aligned_pointer_as_index %v_407 : memref<?xindex> -> index
    %v_414 = arith.index_cast %v_413 : index to i64
    %v_415 = llvm.inttoptr %v_414 : i64 to !llvm.ptr
    %v_416 = llvm.getelementptr %v_31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_415, %v_416 : !llvm.ptr, !llvm.ptr
    %v_417 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_418 = builtin.unrealized_conversion_cast %v_417 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_419 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_420 = builtin.unrealized_conversion_cast %v_419 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_421 = memref.dim %v_420, %v_233 : memref<?xindex>
    %v_422 = arith.index_cast %v_421 : index to i64
    %v_423 = llvm.getelementptr %v_48[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_422, %v_423 : i64, !llvm.ptr
    %v_424 = memref.extract_aligned_pointer_as_index %v_418 : memref<?xindex> -> index
    %v_425 = arith.index_cast %v_424 : index to i64
    %v_426 = llvm.inttoptr %v_425 : i64 to !llvm.ptr
    %v_427 = llvm.getelementptr %v_48[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_426, %v_427 : !llvm.ptr, !llvm.ptr
    %v_428 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_429 = builtin.unrealized_conversion_cast %v_428 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_430 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_431 = builtin.unrealized_conversion_cast %v_430 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_432 = memref.dim %v_431, %v_233 : memref<?xf64>
    %v_433 = arith.index_cast %v_432 : index to i64
    %v_434 = llvm.getelementptr %v_93[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_433, %v_434 : i64, !llvm.ptr
    %v_435 = memref.extract_aligned_pointer_as_index %v_429 : memref<?xf64> -> index
    %v_436 = arith.index_cast %v_435 : index to i64
    %v_437 = llvm.inttoptr %v_436 : i64 to !llvm.ptr
    %v_438 = llvm.getelementptr %v_93[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_437, %v_438 : !llvm.ptr, !llvm.ptr
    %v_439 = llvm.load %v_128 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_440 = builtin.unrealized_conversion_cast %v_439 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_441 = llvm.load %v_128 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_442 = builtin.unrealized_conversion_cast %v_441 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_443 = memref.dim %v_442, %v_233 : memref<?xindex>
    %v_444 = arith.index_cast %v_443 : index to i64
    %v_445 = llvm.getelementptr %v_115[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_444, %v_445 : i64, !llvm.ptr
    %v_446 = memref.extract_aligned_pointer_as_index %v_440 : memref<?xindex> -> index
    %v_447 = arith.index_cast %v_446 : index to i64
    %v_448 = llvm.inttoptr %v_447 : i64 to !llvm.ptr
    %v_449 = llvm.getelementptr %v_115[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_448, %v_449 : !llvm.ptr, !llvm.ptr
    %v_450 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_451 = builtin.unrealized_conversion_cast %v_450 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_452 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_453 = builtin.unrealized_conversion_cast %v_452 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_454 = memref.dim %v_453, %v_233 : memref<?xindex>
    %v_455 = arith.index_cast %v_454 : index to i64
    %v_456 = llvm.getelementptr %v_132[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_455, %v_456 : i64, !llvm.ptr
    %v_457 = memref.extract_aligned_pointer_as_index %v_451 : memref<?xindex> -> index
    %v_458 = arith.index_cast %v_457 : index to i64
    %v_459 = llvm.inttoptr %v_458 : i64 to !llvm.ptr
    %v_460 = llvm.getelementptr %v_132[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_459, %v_460 : !llvm.ptr, !llvm.ptr
    %v_461 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_462 = builtin.unrealized_conversion_cast %v_461 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_463 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_464 = builtin.unrealized_conversion_cast %v_463 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_465 = memref.dim %v_464, %v_233 : memref<?xf64>
    %v_466 = arith.index_cast %v_465 : index to i64
    %v_467 = llvm.getelementptr %v_172[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_466, %v_467 : i64, !llvm.ptr
    %v_468 = memref.extract_aligned_pointer_as_index %v_462 : memref<?xf64> -> index
    %v_469 = arith.index_cast %v_468 : index to i64
    %v_470 = llvm.inttoptr %v_469 : i64 to !llvm.ptr
    %v_471 = llvm.getelementptr %v_172[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_470, %v_471 : !llvm.ptr, !llvm.ptr
    %v_472 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_473 = builtin.unrealized_conversion_cast %v_472 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_474 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_475 = builtin.unrealized_conversion_cast %v_474 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_476 = memref.dim %v_475, %v_233 : memref<?xf64>
    %v_477 = arith.index_cast %v_476 : index to i64
    %v_478 = llvm.getelementptr %v_203[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_477, %v_478 : i64, !llvm.ptr
    %v_479 = memref.extract_aligned_pointer_as_index %v_473 : memref<?xf64> -> index
    %v_480 = arith.index_cast %v_479 : index to i64
    %v_481 = llvm.inttoptr %v_480 : i64 to !llvm.ptr
    %v_482 = llvm.getelementptr %v_203[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_481, %v_482 : !llvm.ptr, !llvm.ptr
    %v_483 = llvm.mlir.undef : !llvm.struct<(!llvm.ptr)>
    %v_484 = llvm.insertvalue %_A_7, %v_483[0] : !llvm.struct<(!llvm.ptr)>
    llvm.store %v_484, %_ret : !llvm.struct<(!llvm.ptr)>, !llvm.ptr
    func.return
  }
}
