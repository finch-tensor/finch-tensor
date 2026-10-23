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
    %v_62 = llvm.getelementptr %v[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_63 = llvm.load %v_62 : !llvm.ptr -> i64
    %v_64 = arith.index_cast %v_63 : i64 to index
    %v_65 = llvm.extractvalue %_A_16[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>
    %v_66 = llvm.getelementptr %v_65[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_67 = llvm.load %v_66 : !llvm.ptr -> !llvm.ptr
    %v_68 = llvm.getelementptr %v_67[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
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
    %v_87 = llvm.getelementptr %v_86[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_88 = llvm.load %v_87 : !llvm.ptr -> i64
    %v_89 = arith.index_cast %v_88 : i64 to index
    %v_90 = llvm.getelementptr %v_65[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_91 = llvm.load %v_90 : !llvm.ptr -> !llvm.ptr
    %v_92 = llvm.getelementptr %v_91[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
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
    %v_104 = builtin.unrealized_conversion_cast %v_103 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_105 = builtin.unrealized_conversion_cast %v_104 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_106 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_105, %v_106 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_107 = llvm.getelementptr %v_65[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_108 = llvm.load %v_107 : !llvm.ptr -> !llvm.ptr
    %v_109 = llvm.getelementptr %v_108[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_110 = llvm.load %v_109 : !llvm.ptr -> !llvm.ptr
    %v_111 = llvm.getelementptr %v_110[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_112 = llvm.load %v_111 : !llvm.ptr -> !llvm.ptr
    %v_113 = llvm.getelementptr %v_110[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_114 = llvm.load %v_113 : !llvm.ptr -> i64
    %v_115 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_116 = llvm.insertvalue %v_112, %v_115[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_117 = llvm.insertvalue %v_112, %v_116[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_118 = llvm.insertvalue %v_13, %v_117[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_119 = llvm.insertvalue %v_114, %v_118[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_120 = llvm.insertvalue %v_14, %v_119[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_121 = builtin.unrealized_conversion_cast %v_120 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_122 = builtin.unrealized_conversion_cast %v_121 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_123 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_122, %v_123 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_124 = llvm.getelementptr %v_65[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_125 = llvm.load %v_124 : !llvm.ptr -> i64
    %v_126 = arith.index_cast %v_125 : i64 to index
    %v_127 = llvm.getelementptr %_A_18[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_128 = llvm.load %v_127 : !llvm.ptr -> !llvm.ptr
    %v_129 = llvm.getelementptr %v_128[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_130 = llvm.load %v_129 : !llvm.ptr -> !llvm.ptr
    %v_131 = llvm.getelementptr %v_128[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_132 = llvm.load %v_131 : !llvm.ptr -> i64
    %v_133 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_134 = llvm.insertvalue %v_130, %v_133[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_135 = llvm.insertvalue %v_130, %v_134[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_136 = llvm.insertvalue %v_13, %v_135[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_137 = llvm.insertvalue %v_132, %v_136[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_138 = llvm.insertvalue %v_14, %v_137[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_139 = builtin.unrealized_conversion_cast %v_138 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_140 = builtin.unrealized_conversion_cast %v_139 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_141 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_140, %v_141 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_142 = llvm.getelementptr %_A_18[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_143 = llvm.load %v_142 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_144 = llvm.extractvalue %v_143[1] : !llvm.struct<(i64, i64)>
    %v_145 = arith.index_cast %v_144 : i64 to index
    %v_146 = llvm.getelementptr %_A_18[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_147 = llvm.load %v_146 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_148 = llvm.extractvalue %v_147[0] : !llvm.struct<(i64, i64)>
    %v_149 = arith.index_cast %v_148 : i64 to index
    %v_150 = llvm.getelementptr %_A_18[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_151 = llvm.load %v_150 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_152 = llvm.extractvalue %v_151[1] : !llvm.struct<(i64, i64)>
    %v_153 = arith.index_cast %v_152 : i64 to index
    %v_154 = llvm.getelementptr %_A_7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_155 = llvm.load %v_154 : !llvm.ptr -> !llvm.ptr
    %v_156 = llvm.getelementptr %v_155[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_157 = llvm.load %v_156 : !llvm.ptr -> !llvm.ptr
    %v_158 = llvm.getelementptr %v_155[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_159 = llvm.load %v_158 : !llvm.ptr -> i64
    %v_160 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_161 = llvm.insertvalue %v_157, %v_160[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_162 = llvm.insertvalue %v_157, %v_161[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_163 = llvm.insertvalue %v_13, %v_162[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_164 = llvm.insertvalue %v_159, %v_163[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_165 = llvm.insertvalue %v_14, %v_164[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_166 = builtin.unrealized_conversion_cast %v_165 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_167 = builtin.unrealized_conversion_cast %v_166 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_168 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_167, %v_168 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_169 = llvm.getelementptr %_A_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_170 = llvm.load %v_169 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_171 = llvm.extractvalue %v_170[0] : !llvm.struct<(i64, i64)>
    %v_172 = arith.index_cast %v_171 : i64 to index
    %v_173 = llvm.getelementptr %_A_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_174 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_175 = llvm.extractvalue %v_174[1] : !llvm.struct<(i64, i64)>
    %v_176 = arith.index_cast %v_175 : i64 to index
    %v_177 = arith.constant 0 : index
    %v_178 = llvm.load %v_141 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_179 = builtin.unrealized_conversion_cast %v_178 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_180 = memref.dim %v_179, %v_177 : memref<?xf64>
    %v_181 = arith.constant 1 : index
    scf.for %v_182 = %v_177 to %v_180 step %v_181 {
      %v_183 = llvm.load %v_141 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_184 = builtin.unrealized_conversion_cast %v_183 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_185 = arith.constant 0.0 : f64
      memref.store %v_185, %v_184[%v_182] : memref<?xf64>
    }
    scf.for %v_186 = %v_177 to %v_27 step %v_181 {
      %v_187 = arith.muli %v_149, %v_186 : index
      scf.for %v_188 = %v_177 to %v_64 step %v_181 {
        %v_189 = arith.muli %v_153, %v_188 : index
        %v_190 = arith.addi %v_187, %v_189 : index
        %v_191 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_192 = builtin.unrealized_conversion_cast %v_191 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_193 = memref.load %v_192[%v_188] : memref<?xindex>
        %v_194 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_195 = builtin.unrealized_conversion_cast %v_194 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_196 = arith.addi %v_181, %v_188 : index
        %v_197 = memref.load %v_195[%v_196] : memref<?xindex>
        %v_198 = arith.cmpi slt, %v_193, %v_197 : index
        %v_206, %v_207 = scf.if %v_198 -> (index, index) {
          %v_199 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_200 = builtin.unrealized_conversion_cast %v_199 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_201 = memref.load %v_200[%v_193] : memref<?xindex>
          %v_202 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_203 = builtin.unrealized_conversion_cast %v_202 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_204 = arith.subi %v_197, %v_181 : index
          %v_205 = memref.load %v_203[%v_204] : memref<?xindex>
          scf.yield %v_201, %v_205 : index, index
        } else {
          scf.yield %v_181, %v_177 : index, index
        }
        %v_208 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_209 = builtin.unrealized_conversion_cast %v_208 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_210 = memref.load %v_209[%v_193] : memref<?xindex>
        %v_211 = arith.cmpi slt, %v_210, %v_177 : index
        %v_216 = scf.if %v_211 -> (index) {
          %v_212 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_213 = builtin.unrealized_conversion_cast %v_212 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_214 = arith.subi %v_197, %v_181 : index
          %v_215 = func.call @scansearch_index(%v_213, %v_177, %v_193, %v_214) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_215 : index
        } else {
          scf.yield %v_193 : index
        }
        %v_243:3 = scf.while (%v_217 = %v_177, %v_218 = %v_216, %v_219 = %v_206) : (index, index, index) -> (index, index, index) {
          %v_220 = arith.addi %v_181, %v_207 : index
          %v_221 = arith.minsi %v_27, %v_220 : index
          %v_222 = arith.cmpi slt, %v_219, %v_221 : index
          scf.condition(%v_222) %v_217, %v_218, %v_219 : index, index, index
        } do {
          ^bb(%v_217: index, %v_218: index, %v_219: index):
          %v_223 = arith.constant 1 : i1
          %v_240, %v_241, %v_242 = scf.if %v_223 -> (index, index, index) {
            %v_224 = arith.maxsi %v_217, %v_219 : index
            %v_225 = arith.addi %v_181, %v_219 : index
            scf.for %v_226 = %v_224 to %v_225 step %v_181 {
              %v_227 = arith.cmpi eq, %v_226, %v_186 : index
              scf.if %v_227 {
                %v_228 = llvm.load %v_141 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
                %v_229 = builtin.unrealized_conversion_cast %v_228 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
                %v_230 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
                %v_231 = builtin.unrealized_conversion_cast %v_230 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
                %v_232 = memref.load %v_231[%v_218] : memref<?xf64>
                memref.store %v_232, %v_229[%v_190] : memref<?xf64>
              }
            }
            %v_233 = arith.addi %v_181, %v_219 : index
            %v_234 = arith.addi %v_181, %v_218 : index
            %v_235 = arith.cmpi slt, %v_234, %v_197 : index
            %v_239 = scf.if %v_235 -> (index) {
              %v_236 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_237 = builtin.unrealized_conversion_cast %v_236 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
              %v_238 = memref.load %v_237[%v_234] : memref<?xindex>
              scf.yield %v_238 : index
            } else {
              scf.yield %v_27 : index
            }
            scf.yield %v_233, %v_234, %v_239 : index, index, index
          } else {
            scf.yield %v_217, %v_218, %v_219 : index, index, index
          }
          scf.yield %v_240, %v_241, %v_242 : index, index, index
        }
      }
    }
    %v_244 = llvm.load %v_168 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_245 = builtin.unrealized_conversion_cast %v_244 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_246 = memref.dim %v_245, %v_177 : memref<?xf64>
    scf.for %v_247 = %v_177 to %v_246 step %v_181 {
      %v_248 = llvm.load %v_168 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_249 = builtin.unrealized_conversion_cast %v_248 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_250 = arith.constant 0.0 : f64
      memref.store %v_250, %v_249[%v_247] : memref<?xf64>
    }
    scf.for %v_251 = %v_177 to %v_126 step %v_181 {
      %v_252 = arith.muli %v_149, %v_251 : index
      scf.for %v_253 = %v_177 to %v_145 step %v_181 {
        %v_254 = arith.muli %v_172, %v_253 : index
        %v_255 = arith.muli %v_153, %v_253 : index
        %v_256 = arith.addi %v_252, %v_255 : index
        %v_257 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_258 = builtin.unrealized_conversion_cast %v_257 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_259 = memref.load %v_258[%v_251] : memref<?xindex>
        %v_260 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_261 = builtin.unrealized_conversion_cast %v_260 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_262 = arith.addi %v_181, %v_251 : index
        %v_263 = memref.load %v_261[%v_262] : memref<?xindex>
        %v_264 = arith.cmpi slt, %v_259, %v_263 : index
        %v_272, %v_273 = scf.if %v_264 -> (index, index) {
          %v_265 = llvm.load %v_123 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_266 = builtin.unrealized_conversion_cast %v_265 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_267 = memref.load %v_266[%v_259] : memref<?xindex>
          %v_268 = llvm.load %v_123 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_269 = builtin.unrealized_conversion_cast %v_268 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_270 = arith.subi %v_263, %v_181 : index
          %v_271 = memref.load %v_269[%v_270] : memref<?xindex>
          scf.yield %v_267, %v_271 : index, index
        } else {
          scf.yield %v_181, %v_177 : index, index
        }
        %v_274 = llvm.load %v_123 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_275 = builtin.unrealized_conversion_cast %v_274 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_276 = memref.load %v_275[%v_259] : memref<?xindex>
        %v_277 = arith.cmpi slt, %v_276, %v_177 : index
        %v_282 = scf.if %v_277 -> (index) {
          %v_278 = llvm.load %v_123 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_279 = builtin.unrealized_conversion_cast %v_278 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_280 = arith.subi %v_263, %v_181 : index
          %v_281 = func.call @scansearch_index(%v_279, %v_177, %v_259, %v_280) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_281 : index
        } else {
          scf.yield %v_259 : index
        }
        %v_328:3 = scf.while (%v_283 = %v_177, %v_284 = %v_282, %v_285 = %v_272) : (index, index, index) -> (index, index, index) {
          %v_286 = arith.addi %v_181, %v_273 : index
          %v_287 = arith.minsi %v_89, %v_286 : index
          %v_288 = arith.cmpi slt, %v_285, %v_287 : index
          scf.condition(%v_288) %v_283, %v_284, %v_285 : index, index, index
        } do {
          ^bb_2(%v_283: index, %v_284: index, %v_285: index):
          %v_289 = arith.constant 1 : i1
          %v_325, %v_326, %v_327 = scf.if %v_289 -> (index, index, index) {
            %v_290 = arith.addi %v_181, %v_285 : index
            %v_291 = arith.minsi %v_290, %v_285 : index
            scf.for %v_292 = %v_283 to %v_291 step %v_181 {
              %v_293 = arith.muli %v_176, %v_292 : index
              %v_294 = arith.addi %v_254, %v_293 : index
              %v_295 = llvm.load %v_168 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_296 = builtin.unrealized_conversion_cast %v_295 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_297 = llvm.load %v_168 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_298 = builtin.unrealized_conversion_cast %v_297 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_299 = memref.load %v_298[%v_294] : memref<?xf64>
              memref.store %v_299, %v_296[%v_294] : memref<?xf64>
            }
            %v_300 = arith.maxsi %v_283, %v_285 : index
            %v_301 = arith.addi %v_181, %v_285 : index
            scf.for %v_302 = %v_300 to %v_301 step %v_181 {
              %v_303 = arith.muli %v_176, %v_302 : index
              %v_304 = arith.addi %v_254, %v_303 : index
              %v_305 = llvm.load %v_168 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_306 = builtin.unrealized_conversion_cast %v_305 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_307 = llvm.load %v_168 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_308 = builtin.unrealized_conversion_cast %v_307 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_309 = memref.load %v_308[%v_304] : memref<?xf64>
              %v_310 = llvm.load %v_141 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_311 = builtin.unrealized_conversion_cast %v_310 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_312 = memref.load %v_311[%v_256] : memref<?xf64>
              %v_313 = llvm.load %v_84 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_314 = builtin.unrealized_conversion_cast %v_313 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_315 = memref.load %v_314[%v_284] : memref<?xf64>
              %v_316 = arith.mulf %v_312, %v_315 : f64
              %v_317 = arith.addf %v_309, %v_316 : f64
              memref.store %v_317, %v_306[%v_304] : memref<?xf64>
            }
            %v_318 = arith.addi %v_181, %v_285 : index
            %v_319 = arith.addi %v_181, %v_284 : index
            %v_320 = arith.cmpi slt, %v_319, %v_263 : index
            %v_324 = scf.if %v_320 -> (index) {
              %v_321 = llvm.load %v_123 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_322 = builtin.unrealized_conversion_cast %v_321 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
              %v_323 = memref.load %v_322[%v_319] : memref<?xindex>
              scf.yield %v_323 : index
            } else {
              scf.yield %v_89 : index
            }
            scf.yield %v_318, %v_319, %v_324 : index, index, index
          } else {
            scf.yield %v_283, %v_284, %v_285 : index, index, index
          }
          scf.yield %v_325, %v_326, %v_327 : index, index, index
        }
        %v_329 = arith.addi %v_181, %v_273 : index
        %v_330 = arith.maxsi %v_177, %v_329 : index
        scf.for %v_331 = %v_330 to %v_89 step %v_181 {
          %v_332 = arith.muli %v_176, %v_331 : index
          %v_333 = arith.addi %v_254, %v_332 : index
          %v_334 = llvm.load %v_168 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_335 = builtin.unrealized_conversion_cast %v_334 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_336 = llvm.load %v_168 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_337 = builtin.unrealized_conversion_cast %v_336 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_338 = memref.load %v_337[%v_333] : memref<?xf64>
          memref.store %v_338, %v_335[%v_333] : memref<?xf64>
        }
      }
    }
    %v_339 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_340 = builtin.unrealized_conversion_cast %v_339 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_341 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_342 = builtin.unrealized_conversion_cast %v_341 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_343 = memref.dim %v_342, %v_177 : memref<?xf64>
    %v_344 = arith.index_cast %v_343 : index to i64
    %v_345 = llvm.getelementptr %v_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_344, %v_345 : i64, !llvm.ptr
    %v_346 = memref.extract_aligned_pointer_as_index %v_340 : memref<?xf64> -> index
    %v_347 = arith.index_cast %v_346 : index to i64
    %v_348 = llvm.inttoptr %v_347 : i64 to !llvm.ptr
    %v_349 = llvm.getelementptr %v_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_348, %v_349 : !llvm.ptr, !llvm.ptr
    %v_350 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_351 = builtin.unrealized_conversion_cast %v_350 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_352 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_353 = builtin.unrealized_conversion_cast %v_352 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_354 = memref.dim %v_353, %v_177 : memref<?xindex>
    %v_355 = arith.index_cast %v_354 : index to i64
    %v_356 = llvm.getelementptr %v_31[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_355, %v_356 : i64, !llvm.ptr
    %v_357 = memref.extract_aligned_pointer_as_index %v_351 : memref<?xindex> -> index
    %v_358 = arith.index_cast %v_357 : index to i64
    %v_359 = llvm.inttoptr %v_358 : i64 to !llvm.ptr
    %v_360 = llvm.getelementptr %v_31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_359, %v_360 : !llvm.ptr, !llvm.ptr
    %v_361 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_362 = builtin.unrealized_conversion_cast %v_361 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_363 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_364 = builtin.unrealized_conversion_cast %v_363 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_365 = memref.dim %v_364, %v_177 : memref<?xindex>
    %v_366 = arith.index_cast %v_365 : index to i64
    %v_367 = llvm.getelementptr %v_48[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_366, %v_367 : i64, !llvm.ptr
    %v_368 = memref.extract_aligned_pointer_as_index %v_362 : memref<?xindex> -> index
    %v_369 = arith.index_cast %v_368 : index to i64
    %v_370 = llvm.inttoptr %v_369 : i64 to !llvm.ptr
    %v_371 = llvm.getelementptr %v_48[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_370, %v_371 : !llvm.ptr, !llvm.ptr
    %v_372 = llvm.load %v_84 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_373 = builtin.unrealized_conversion_cast %v_372 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_374 = llvm.load %v_84 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_375 = builtin.unrealized_conversion_cast %v_374 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_376 = memref.dim %v_375, %v_177 : memref<?xf64>
    %v_377 = arith.index_cast %v_376 : index to i64
    %v_378 = llvm.getelementptr %v_71[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_377, %v_378 : i64, !llvm.ptr
    %v_379 = memref.extract_aligned_pointer_as_index %v_373 : memref<?xf64> -> index
    %v_380 = arith.index_cast %v_379 : index to i64
    %v_381 = llvm.inttoptr %v_380 : i64 to !llvm.ptr
    %v_382 = llvm.getelementptr %v_71[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_381, %v_382 : !llvm.ptr, !llvm.ptr
    %v_383 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_384 = builtin.unrealized_conversion_cast %v_383 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_385 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_386 = builtin.unrealized_conversion_cast %v_385 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_387 = memref.dim %v_386, %v_177 : memref<?xindex>
    %v_388 = arith.index_cast %v_387 : index to i64
    %v_389 = llvm.getelementptr %v_93[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_388, %v_389 : i64, !llvm.ptr
    %v_390 = memref.extract_aligned_pointer_as_index %v_384 : memref<?xindex> -> index
    %v_391 = arith.index_cast %v_390 : index to i64
    %v_392 = llvm.inttoptr %v_391 : i64 to !llvm.ptr
    %v_393 = llvm.getelementptr %v_93[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_392, %v_393 : !llvm.ptr, !llvm.ptr
    %v_394 = llvm.load %v_123 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_395 = builtin.unrealized_conversion_cast %v_394 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_396 = llvm.load %v_123 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_397 = builtin.unrealized_conversion_cast %v_396 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_398 = memref.dim %v_397, %v_177 : memref<?xindex>
    %v_399 = arith.index_cast %v_398 : index to i64
    %v_400 = llvm.getelementptr %v_110[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_399, %v_400 : i64, !llvm.ptr
    %v_401 = memref.extract_aligned_pointer_as_index %v_395 : memref<?xindex> -> index
    %v_402 = arith.index_cast %v_401 : index to i64
    %v_403 = llvm.inttoptr %v_402 : i64 to !llvm.ptr
    %v_404 = llvm.getelementptr %v_110[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_403, %v_404 : !llvm.ptr, !llvm.ptr
    %v_405 = llvm.load %v_141 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_406 = builtin.unrealized_conversion_cast %v_405 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_407 = llvm.load %v_141 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_408 = builtin.unrealized_conversion_cast %v_407 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_409 = memref.dim %v_408, %v_177 : memref<?xf64>
    %v_410 = arith.index_cast %v_409 : index to i64
    %v_411 = llvm.getelementptr %v_128[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_410, %v_411 : i64, !llvm.ptr
    %v_412 = memref.extract_aligned_pointer_as_index %v_406 : memref<?xf64> -> index
    %v_413 = arith.index_cast %v_412 : index to i64
    %v_414 = llvm.inttoptr %v_413 : i64 to !llvm.ptr
    %v_415 = llvm.getelementptr %v_128[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_414, %v_415 : !llvm.ptr, !llvm.ptr
    %v_416 = llvm.load %v_168 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_417 = builtin.unrealized_conversion_cast %v_416 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_418 = llvm.load %v_168 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_419 = builtin.unrealized_conversion_cast %v_418 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_420 = memref.dim %v_419, %v_177 : memref<?xf64>
    %v_421 = arith.index_cast %v_420 : index to i64
    %v_422 = llvm.getelementptr %v_155[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_421, %v_422 : i64, !llvm.ptr
    %v_423 = memref.extract_aligned_pointer_as_index %v_417 : memref<?xf64> -> index
    %v_424 = arith.index_cast %v_423 : index to i64
    %v_425 = llvm.inttoptr %v_424 : i64 to !llvm.ptr
    %v_426 = llvm.getelementptr %v_155[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_425, %v_426 : !llvm.ptr, !llvm.ptr
    %v_427 = llvm.mlir.undef : !llvm.struct<(!llvm.ptr)>
    %v_428 = llvm.insertvalue %_A_7, %v_427[0] : !llvm.struct<(!llvm.ptr)>
    llvm.store %v_428, %_ret : !llvm.struct<(!llvm.ptr)>, !llvm.ptr
    func.return
  }
}
