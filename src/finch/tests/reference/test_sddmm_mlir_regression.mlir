Compiling MLIR code:
module {
  func.func @scansearch(
    %arr: memref<?xindex>, %x: index, %lo: index, %hi: index
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

  func.func @main(%_A_19: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>, %_A_20: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>, %_A_21: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>, %_A_9: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_19[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
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
    %v_17 = llvm.extractvalue %_A_19[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_18 = llvm.getelementptr %v_17[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_19 = llvm.load %v_18 : !llvm.ptr -> !llvm.ptr
    %v_20 = llvm.getelementptr %v_17[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_21 = llvm.load %v_20 : !llvm.ptr -> i64
    %v_22 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_23 = llvm.insertvalue %v_19, %v_22[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_24 = llvm.insertvalue %v_19, %v_23[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_25 = llvm.insertvalue %v_7, %v_24[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_26 = llvm.insertvalue %v_21, %v_25[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_27 = llvm.insertvalue %v_8, %v_26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_28 = builtin.unrealized_conversion_cast %v_27 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_29 = builtin.unrealized_conversion_cast %v_28 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_30 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_29, %v_30 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_31 = llvm.extractvalue %_A_19[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_32 = llvm.getelementptr %v_31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_33 = llvm.load %v_32 : !llvm.ptr -> !llvm.ptr
    %v_34 = llvm.getelementptr %v_31[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_35 = llvm.load %v_34 : !llvm.ptr -> i64
    %v_36 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_37 = llvm.insertvalue %v_33, %v_36[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_38 = llvm.insertvalue %v_33, %v_37[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_39 = llvm.insertvalue %v_7, %v_38[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_40 = llvm.insertvalue %v_35, %v_39[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_41 = llvm.insertvalue %v_8, %v_40[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_42 = builtin.unrealized_conversion_cast %v_41 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_43 = builtin.unrealized_conversion_cast %v_42 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_44 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_43, %v_44 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_45 = llvm.extractvalue %_A_19[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_46 = arith.index_cast %v_45 : i64 to index
    %v_47 = llvm.extractvalue %_A_19[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_48 = arith.index_cast %v_47 : i64 to index
    %v_49 = llvm.extractvalue %_A_20[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_50 = llvm.getelementptr %v_49[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_51 = llvm.load %v_50 : !llvm.ptr -> !llvm.ptr
    %v_52 = llvm.getelementptr %v_49[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_53 = llvm.load %v_52 : !llvm.ptr -> i64
    %v_54 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_55 = llvm.insertvalue %v_51, %v_54[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_56 = llvm.insertvalue %v_51, %v_55[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_57 = llvm.insertvalue %v_7, %v_56[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_58 = llvm.insertvalue %v_53, %v_57[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_59 = llvm.insertvalue %v_8, %v_58[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_60 = builtin.unrealized_conversion_cast %v_59 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_61 = builtin.unrealized_conversion_cast %v_60 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_62 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_61, %v_62 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_63 = llvm.extractvalue %_A_20[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_64 = arith.index_cast %v_63 : i64 to index
    %v_65 = llvm.extractvalue %_A_20[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_66 = arith.index_cast %v_65 : i64 to index
    %v_67 = llvm.extractvalue %_A_21[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_68 = llvm.getelementptr %v_67[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_69 = llvm.load %v_68 : !llvm.ptr -> !llvm.ptr
    %v_70 = llvm.getelementptr %v_67[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_71 = llvm.load %v_70 : !llvm.ptr -> i64
    %v_72 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_73 = llvm.insertvalue %v_69, %v_72[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_74 = llvm.insertvalue %v_69, %v_73[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_75 = llvm.insertvalue %v_7, %v_74[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_76 = llvm.insertvalue %v_71, %v_75[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_77 = llvm.insertvalue %v_8, %v_76[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_78 = builtin.unrealized_conversion_cast %v_77 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_79 = builtin.unrealized_conversion_cast %v_78 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_80 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_79, %v_80 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_81 = llvm.extractvalue %_A_21[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_82 = arith.index_cast %v_81 : i64 to index
    %v_83 = llvm.extractvalue %_A_21[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_84 = arith.index_cast %v_83 : i64 to index
    %v_85 = llvm.extractvalue %_A_9[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_86 = llvm.getelementptr %v_85[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_87 = llvm.load %v_86 : !llvm.ptr -> !llvm.ptr
    %v_88 = llvm.getelementptr %v_85[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_89 = llvm.load %v_88 : !llvm.ptr -> i64
    %v_90 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_91 = llvm.insertvalue %v_87, %v_90[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_92 = llvm.insertvalue %v_87, %v_91[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_93 = llvm.insertvalue %v_7, %v_92[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_94 = llvm.insertvalue %v_89, %v_93[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_95 = llvm.insertvalue %v_8, %v_94[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_96 = builtin.unrealized_conversion_cast %v_95 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_97 = builtin.unrealized_conversion_cast %v_96 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_98 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_97, %v_98 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_99 = llvm.extractvalue %_A_9[1, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_100 = arith.index_cast %v_99 : i64 to index
    %v_101 = llvm.extractvalue %_A_9[1, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_102 = arith.index_cast %v_101 : i64 to index
    %v_103 = arith.constant 0 : index
    %v_104 = llvm.load %v_98 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_105 = builtin.unrealized_conversion_cast %v_104 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_106 = memref.dim %v_105, %v_103 : memref<?xf64>
    %v_107 = arith.constant 1 : index
    scf.for %v_108 = %v_103 to %v_106 step %v_107 {
      %v_109 = llvm.load %v_98 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_110 = builtin.unrealized_conversion_cast %v_109 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_111 = arith.constant 0.0 : f64
      memref.store %v_111, %v_110[%v_108] : memref<?xf64>
    }
    scf.for %v_112 = %v_103 to %v_46 step %v_107 {
      %v_113 = llvm.extractvalue %_A_9[2, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_114 = arith.index_cast %v_113 : i64 to index
      %v_115 = arith.muli %v_114, %v_112 : index
      scf.for %v_116 = %v_103 to %v_66 step %v_107 {
        %v_117 = arith.muli %v_112, %v_66 : index
        %v_118 = arith.addi %v_117, %v_116 : index
        %v_119 = llvm.load %v_30 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_120 = builtin.unrealized_conversion_cast %v_119 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_121 = memref.load %v_120[%v_112] : memref<?xindex>
        %v_122 = llvm.load %v_30 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_123 = builtin.unrealized_conversion_cast %v_122 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_124 = arith.addi %v_107, %v_112 : index
        %v_125 = memref.load %v_123[%v_124] : memref<?xindex>
        %v_126 = arith.cmpi slt, %v_121, %v_125 : index
        %v_134, %v_135 = scf.if %v_126 -> (index, index) {
          %v_127 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_128 = builtin.unrealized_conversion_cast %v_127 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_129 = memref.load %v_128[%v_121] : memref<?xindex>
          %v_130 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_131 = builtin.unrealized_conversion_cast %v_130 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_132 = arith.subi %v_125, %v_107 : index
          %v_133 = memref.load %v_131[%v_132] : memref<?xindex>
          scf.yield %v_129, %v_133 : index, index
        } else {
          scf.yield %v_107, %v_103 : index, index
        }
        %v_136 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_137 = builtin.unrealized_conversion_cast %v_136 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_138 = memref.load %v_137[%v_121] : memref<?xindex>
        %v_139 = arith.cmpi slt, %v_138, %v_103 : index
        %v_144 = scf.if %v_139 -> (index) {
          %v_140 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_141 = builtin.unrealized_conversion_cast %v_140 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_142 = arith.subi %v_125, %v_107 : index
          %v_143 = func.call @scansearch(%v_141, %v_103, %v_121, %v_142) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_143 : index
        } else {
          scf.yield %v_121 : index
        }
        %v_196:3 = scf.while (%v_145 = %v_103, %v_146 = %v_144, %v_147 = %v_134) : (index, index, index) -> (index, index, index) {
          %v_148 = arith.addi %v_107, %v_135 : index
          %v_149 = arith.minsi %v_48, %v_148 : index
          %v_150 = arith.cmpi slt, %v_147, %v_149 : index
          scf.condition(%v_150) %v_145, %v_146, %v_147 : index, index, index
        } do {
          ^bb(%v_145: index, %v_146: index, %v_147: index):
          %v_151 = arith.addi %v_107, %v_147 : index
          %v_152 = arith.minsi %v_151, %v_147 : index
          scf.for %v_153 = %v_145 to %v_152 step %v_107 {
            %v_154 = llvm.extractvalue %_A_9[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_155 = arith.index_cast %v_154 : i64 to index
            %v_156 = arith.muli %v_155, %v_153 : index
            %v_157 = arith.addi %v_115, %v_156 : index
            %v_158 = llvm.load %v_98 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_159 = builtin.unrealized_conversion_cast %v_158 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_160 = llvm.load %v_98 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_161 = builtin.unrealized_conversion_cast %v_160 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_162 = memref.load %v_161[%v_157] : memref<?xf64>
            memref.store %v_162, %v_159[%v_157] : memref<?xf64>
          }
          %v_163 = arith.maxsi %v_145, %v_147 : index
          %v_164 = arith.addi %v_107, %v_147 : index
          scf.for %v_165 = %v_163 to %v_164 step %v_107 {
            %v_166 = llvm.extractvalue %_A_9[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_167 = arith.index_cast %v_166 : i64 to index
            %v_168 = arith.muli %v_167, %v_165 : index
            %v_169 = arith.addi %v_115, %v_168 : index
            %v_170 = arith.muli %v_116, %v_84 : index
            %v_171 = arith.addi %v_170, %v_165 : index
            %v_172 = llvm.load %v_98 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_173 = builtin.unrealized_conversion_cast %v_172 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_174 = llvm.load %v_98 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_175 = builtin.unrealized_conversion_cast %v_174 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_176 = memref.load %v_175[%v_169] : memref<?xf64>
            %v_177 = llvm.load %v_16 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_178 = builtin.unrealized_conversion_cast %v_177 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_179 = memref.load %v_178[%v_146] : memref<?xf64>
            %v_180 = llvm.load %v_62 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_181 = builtin.unrealized_conversion_cast %v_180 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_182 = memref.load %v_181[%v_118] : memref<?xf64>
            %v_183 = arith.mulf %v_179, %v_182 : f64
            %v_184 = llvm.load %v_80 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_185 = builtin.unrealized_conversion_cast %v_184 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_186 = memref.load %v_185[%v_171] : memref<?xf64>
            %v_187 = arith.mulf %v_183, %v_186 : f64
            %v_188 = arith.addf %v_176, %v_187 : f64
            memref.store %v_188, %v_173[%v_169] : memref<?xf64>
          }
          %v_189 = arith.addi %v_107, %v_147 : index
          %v_190 = arith.addi %v_107, %v_146 : index
          %v_191 = arith.cmpi slt, %v_190, %v_125 : index
          %v_195 = scf.if %v_191 -> (index) {
            %v_192 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_193 = builtin.unrealized_conversion_cast %v_192 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
            %v_194 = memref.load %v_193[%v_190] : memref<?xindex>
            scf.yield %v_194 : index
          } else {
            scf.yield %v_48 : index
          }
          scf.yield %v_189, %v_190, %v_195 : index, index, index
        }
        %v_197 = arith.addi %v_107, %v_135 : index
        %v_198 = arith.maxsi %v_103, %v_197 : index
        scf.for %v_199 = %v_198 to %v_48 step %v_107 {
          %v_200 = llvm.extractvalue %_A_9[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_201 = arith.index_cast %v_200 : i64 to index
          %v_202 = arith.muli %v_201, %v_199 : index
          %v_203 = arith.addi %v_115, %v_202 : index
          %v_204 = llvm.load %v_98 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_205 = builtin.unrealized_conversion_cast %v_204 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_206 = llvm.load %v_98 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_207 = builtin.unrealized_conversion_cast %v_206 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_208 = memref.load %v_207[%v_203] : memref<?xf64>
          memref.store %v_208, %v_205[%v_203] : memref<?xf64>
        }
      }
    }
    %v_209 = llvm.load %v_16 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_210 = builtin.unrealized_conversion_cast %v_209 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_211 = llvm.load %v_16 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_212 = builtin.unrealized_conversion_cast %v_211 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_213 = memref.dim %v_212, %v_103 : memref<?xf64>
    %v_214 = arith.index_cast %v_213 : index to i64
    %v_215 = llvm.getelementptr %v[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_214, %v_215 : i64, !llvm.ptr
    %v_216 = memref.extract_aligned_pointer_as_index %v_210 : memref<?xf64> -> index
    %v_217 = arith.index_cast %v_216 : index to i64
    %v_218 = llvm.inttoptr %v_217 : i64 to !llvm.ptr
    %v_219 = llvm.getelementptr %v[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_218, %v_219 : !llvm.ptr, !llvm.ptr
    %v_220 = llvm.load %v_30 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_221 = builtin.unrealized_conversion_cast %v_220 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_222 = llvm.load %v_30 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_223 = builtin.unrealized_conversion_cast %v_222 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_224 = memref.dim %v_223, %v_103 : memref<?xindex>
    %v_225 = arith.index_cast %v_224 : index to i64
    %v_226 = llvm.getelementptr %v_17[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_225, %v_226 : i64, !llvm.ptr
    %v_227 = memref.extract_aligned_pointer_as_index %v_221 : memref<?xindex> -> index
    %v_228 = arith.index_cast %v_227 : index to i64
    %v_229 = llvm.inttoptr %v_228 : i64 to !llvm.ptr
    %v_230 = llvm.getelementptr %v_17[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_229, %v_230 : !llvm.ptr, !llvm.ptr
    %v_231 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_232 = builtin.unrealized_conversion_cast %v_231 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_233 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_234 = builtin.unrealized_conversion_cast %v_233 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_235 = memref.dim %v_234, %v_103 : memref<?xindex>
    %v_236 = arith.index_cast %v_235 : index to i64
    %v_237 = llvm.getelementptr %v_31[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_236, %v_237 : i64, !llvm.ptr
    %v_238 = memref.extract_aligned_pointer_as_index %v_232 : memref<?xindex> -> index
    %v_239 = arith.index_cast %v_238 : index to i64
    %v_240 = llvm.inttoptr %v_239 : i64 to !llvm.ptr
    %v_241 = llvm.getelementptr %v_31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_240, %v_241 : !llvm.ptr, !llvm.ptr
    %v_242 = llvm.load %v_62 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_243 = builtin.unrealized_conversion_cast %v_242 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_244 = llvm.load %v_62 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_245 = builtin.unrealized_conversion_cast %v_244 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_246 = memref.dim %v_245, %v_103 : memref<?xf64>
    %v_247 = arith.index_cast %v_246 : index to i64
    %v_248 = llvm.getelementptr %v_49[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_247, %v_248 : i64, !llvm.ptr
    %v_249 = memref.extract_aligned_pointer_as_index %v_243 : memref<?xf64> -> index
    %v_250 = arith.index_cast %v_249 : index to i64
    %v_251 = llvm.inttoptr %v_250 : i64 to !llvm.ptr
    %v_252 = llvm.getelementptr %v_49[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_251, %v_252 : !llvm.ptr, !llvm.ptr
    %v_253 = llvm.load %v_80 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_254 = builtin.unrealized_conversion_cast %v_253 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_255 = llvm.load %v_80 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_256 = builtin.unrealized_conversion_cast %v_255 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_257 = memref.dim %v_256, %v_103 : memref<?xf64>
    %v_258 = arith.index_cast %v_257 : index to i64
    %v_259 = llvm.getelementptr %v_67[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_258, %v_259 : i64, !llvm.ptr
    %v_260 = memref.extract_aligned_pointer_as_index %v_254 : memref<?xf64> -> index
    %v_261 = arith.index_cast %v_260 : index to i64
    %v_262 = llvm.inttoptr %v_261 : i64 to !llvm.ptr
    %v_263 = llvm.getelementptr %v_67[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_262, %v_263 : !llvm.ptr, !llvm.ptr
    %v_264 = llvm.load %v_98 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_265 = builtin.unrealized_conversion_cast %v_264 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_266 = llvm.load %v_98 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_267 = builtin.unrealized_conversion_cast %v_266 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_268 = memref.dim %v_267, %v_103 : memref<?xf64>
    %v_269 = arith.index_cast %v_268 : index to i64
    %v_270 = llvm.getelementptr %v_85[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_269, %v_270 : i64, !llvm.ptr
    %v_271 = memref.extract_aligned_pointer_as_index %v_265 : memref<?xf64> -> index
    %v_272 = arith.index_cast %v_271 : index to i64
    %v_273 = llvm.inttoptr %v_272 : i64 to !llvm.ptr
    %v_274 = llvm.getelementptr %v_85[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_273, %v_274 : !llvm.ptr, !llvm.ptr
    %v_275 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_276 = llvm.insertvalue %_A_9, %v_275[0] : !llvm.struct<(!llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_276, %_ret : !llvm.struct<(!llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}
