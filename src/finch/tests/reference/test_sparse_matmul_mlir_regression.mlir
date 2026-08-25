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

  func.func @main(%_A_15: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_16: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_18: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_A_7: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_15[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_2 = builtin.unrealized_conversion_cast %v : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_3 = llvm.extractvalue %_A_15[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_4 = builtin.unrealized_conversion_cast %v_3 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_5 = llvm.extractvalue %_A_15[0, 0, 4] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_6 = builtin.unrealized_conversion_cast %v_5 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_7 = llvm.extractvalue %_A_15[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_8 = arith.index_cast %v_7 : i64 to index
    %v_9 = llvm.extractvalue %_A_15[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_10 = arith.index_cast %v_9 : i64 to index
    %v_11 = llvm.extractvalue %_A_16[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_12 = builtin.unrealized_conversion_cast %v_11 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_13 = llvm.extractvalue %_A_16[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_14 = builtin.unrealized_conversion_cast %v_13 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_15 = llvm.extractvalue %_A_16[0, 0, 4] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_16 = builtin.unrealized_conversion_cast %v_15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_17 = llvm.extractvalue %_A_16[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_18 = arith.index_cast %v_17 : i64 to index
    %v_19 = llvm.extractvalue %_A_16[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_20 = arith.index_cast %v_19 : i64 to index
    %v_21 = llvm.extractvalue %_A_18[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_22 = builtin.unrealized_conversion_cast %v_21 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_23 = llvm.extractvalue %_A_18[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_24 = arith.index_cast %v_23 : i64 to index
    %v_25 = llvm.extractvalue %_A_18[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_26 = arith.index_cast %v_25 : i64 to index
    %v_27 = llvm.extractvalue %_A_7[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_28 = builtin.unrealized_conversion_cast %v_27 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_29 = llvm.extractvalue %_A_7[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_30 = arith.index_cast %v_29 : i64 to index
    %v_31 = llvm.extractvalue %_A_7[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_32 = arith.index_cast %v_31 : i64 to index
    %v_33 = arith.constant 0 : index
    %v_34 = memref.dim %v_22, %v_33 : memref<?xf64>
    %v_35 = arith.constant 1 : index
    scf.for %v_36 = %v_33 to %v_34 step %v_35 {
      %v_37 = arith.constant 0.0 : f64
      memref.store %v_37, %v_22[%v_36] : memref<?xf64>
    }
    scf.for %v_38 = %v_33 to %v_8 step %v_35 {
      %v_39 = llvm.extractvalue %_A_15[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_40 = arith.index_cast %v_39 : i64 to index
      %v_41 = arith.muli %v_40, %v_38 : index
      %v_42 = memref.load %v_4[%v_41] : memref<?xindex>
      %v_43 = arith.addi %v_35, %v_41 : index
      %v_44 = memref.load %v_4[%v_43] : memref<?xindex>
      %v_45 = arith.cmpi slt, %v_42, %v_44 : index
      %v_49, %v_50 = scf.if %v_45 -> (index, index) {
        %v_46 = memref.load %v_6[%v_42] : memref<?xindex>
        %v_47 = arith.subi %v_44, %v_35 : index
        %v_48 = memref.load %v_6[%v_47] : memref<?xindex>
        scf.yield %v_48, %v_46 : index, index
      } else {
        scf.yield %v_33, %v_35 : index, index
      }
      %v_51 = memref.load %v_6[%v_42] : memref<?xindex>
      %v_52 = arith.cmpi slt, %v_51, %v_33 : index
      %v_55 = scf.if %v_52 -> (index) {
        %v_53 = arith.subi %v_44, %v_35 : index
        %v_54 = func.call @scansearch(%v_6, %v_33, %v_42, %v_53) : (memref<?xindex>, index, index, index) -> index
        scf.yield %v_54 : index
      } else {
        scf.yield %v_42 : index
      }
      %v_91:3 = scf.while (%v_56 = %v_33, %v_57 = %v_55, %v_58 = %v_50) : (index, index, index) -> (index, index, index) {
        %v_59 = arith.addi %v_35, %v_49 : index
        %v_60 = arith.minsi %v_10, %v_59 : index
        %v_61 = arith.cmpi slt, %v_58, %v_60 : index
        scf.condition(%v_61) %v_56, %v_57, %v_58 : index, index, index
      } do {
        ^bb(%v_56: index, %v_57: index, %v_58: index):
        %v_62 = arith.addi %v_35, %v_58 : index
        %v_63 = arith.minsi %v_62, %v_58 : index
        scf.for %v_64 = %v_56 to %v_63 step %v_35 {
          %v_65 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_66 = arith.index_cast %v_65 : i64 to index
          %v_67 = arith.muli %v_66, %v_64 : index
          scf.for %v_68 = %v_33 to %v_8 step %v_35 {
            %v_69 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_70 = arith.index_cast %v_69 : i64 to index
            %v_71 = arith.muli %v_70, %v_68 : index
            %v_72 = arith.addi %v_67, %v_71 : index
          }
        }
        %v_73 = arith.maxsi %v_56, %v_58 : index
        %v_74 = arith.addi %v_35, %v_58 : index
        scf.for %v_75 = %v_73 to %v_74 step %v_35 {
          %v_76 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_77 = arith.index_cast %v_76 : i64 to index
          %v_78 = arith.muli %v_77, %v_75 : index
          scf.for %v_79 = %v_33 to %v_8 step %v_35 {
            %v_80 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_81 = arith.index_cast %v_80 : i64 to index
            %v_82 = arith.muli %v_81, %v_79 : index
            %v_83 = arith.addi %v_78, %v_82 : index
            %v_84 = arith.cmpi eq, %v_79, %v_38 : index
            scf.if %v_84 {
              %v_85 = memref.load %v_2[%v_57] : memref<?xf64>
              memref.store %v_85, %v_22[%v_83] : memref<?xf64>
            }
          }
        }
        %v_86 = arith.addi %v_35, %v_58 : index
        %v_87 = arith.addi %v_35, %v_57 : index
        %v_88 = arith.cmpi slt, %v_87, %v_44 : index
        %v_90 = scf.if %v_88 -> (index) {
          %v_89 = memref.load %v_6[%v_87] : memref<?xindex>
          scf.yield %v_89 : index
        } else {
          scf.yield %v_10 : index
        }
        scf.yield %v_86, %v_87, %v_90 : index, index, index
      }
      %v_92 = arith.addi %v_35, %v_49 : index
      %v_93 = arith.maxsi %v_33, %v_92 : index
      scf.for %v_94 = %v_93 to %v_10 step %v_35 {
        %v_95 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_96 = arith.index_cast %v_95 : i64 to index
        %v_97 = arith.muli %v_96, %v_94 : index
        scf.for %v_98 = %v_33 to %v_8 step %v_35 {
          %v_99 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_100 = arith.index_cast %v_99 : i64 to index
          %v_101 = arith.muli %v_100, %v_98 : index
          %v_102 = arith.addi %v_97, %v_101 : index
        }
      }
    }
    %v_103 = memref.dim %v_28, %v_33 : memref<?xf64>
    scf.for %v_104 = %v_33 to %v_103 step %v_35 {
      %v_105 = arith.constant 0.0 : f64
      memref.store %v_105, %v_28[%v_104] : memref<?xf64>
    }
    scf.for %v_106 = %v_33 to %v_18 step %v_35 {
      %v_107 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_108 = arith.index_cast %v_107 : i64 to index
      %v_109 = arith.muli %v_108, %v_106 : index
      %v_110 = llvm.extractvalue %_A_16[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_111 = arith.index_cast %v_110 : i64 to index
      %v_112 = arith.muli %v_111, %v_106 : index
      scf.for %v_113 = %v_33 to %v_26 step %v_35 {
        %v_114 = llvm.extractvalue %_A_7[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_115 = arith.index_cast %v_114 : i64 to index
        %v_116 = arith.muli %v_115, %v_113 : index
        %v_117 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_118 = arith.index_cast %v_117 : i64 to index
        %v_119 = arith.muli %v_118, %v_113 : index
        %v_120 = arith.addi %v_109, %v_119 : index
        %v_121 = memref.load %v_14[%v_112] : memref<?xindex>
        %v_122 = arith.addi %v_35, %v_112 : index
        %v_123 = memref.load %v_14[%v_122] : memref<?xindex>
        %v_124 = arith.cmpi slt, %v_121, %v_123 : index
        %v_128, %v_129 = scf.if %v_124 -> (index, index) {
          %v_125 = memref.load %v_16[%v_121] : memref<?xindex>
          %v_126 = arith.subi %v_123, %v_35 : index
          %v_127 = memref.load %v_16[%v_126] : memref<?xindex>
          scf.yield %v_127, %v_125 : index, index
        } else {
          scf.yield %v_33, %v_35 : index, index
        }
        %v_130 = memref.load %v_16[%v_121] : memref<?xindex>
        %v_131 = arith.cmpi slt, %v_130, %v_33 : index
        %v_134 = scf.if %v_131 -> (index) {
          %v_132 = arith.subi %v_123, %v_35 : index
          %v_133 = func.call @scansearch(%v_16, %v_33, %v_121, %v_132) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_133 : index
        } else {
          scf.yield %v_121 : index
        }
        %v_166:3 = scf.while (%v_135 = %v_33, %v_136 = %v_134, %v_137 = %v_129) : (index, index, index) -> (index, index, index) {
          %v_138 = arith.addi %v_35, %v_128 : index
          %v_139 = arith.minsi %v_20, %v_138 : index
          %v_140 = arith.cmpi slt, %v_137, %v_139 : index
          scf.condition(%v_140) %v_135, %v_136, %v_137 : index, index, index
        } do {
          ^bb_2(%v_135: index, %v_136: index, %v_137: index):
          %v_141 = arith.addi %v_35, %v_137 : index
          %v_142 = arith.minsi %v_141, %v_137 : index
          scf.for %v_143 = %v_135 to %v_142 step %v_35 {
            %v_144 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_145 = arith.index_cast %v_144 : i64 to index
            %v_146 = arith.muli %v_145, %v_143 : index
            %v_147 = arith.addi %v_116, %v_146 : index
            %v_148 = memref.load %v_28[%v_147] : memref<?xf64>
            memref.store %v_148, %v_28[%v_147] : memref<?xf64>
          }
          %v_149 = arith.maxsi %v_135, %v_137 : index
          %v_150 = arith.addi %v_35, %v_137 : index
          scf.for %v_151 = %v_149 to %v_150 step %v_35 {
            %v_152 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_153 = arith.index_cast %v_152 : i64 to index
            %v_154 = arith.muli %v_153, %v_151 : index
            %v_155 = arith.addi %v_116, %v_154 : index
            %v_156 = memref.load %v_28[%v_155] : memref<?xf64>
            %v_157 = memref.load %v_22[%v_120] : memref<?xf64>
            %v_158 = memref.load %v_12[%v_136] : memref<?xf64>
            %v_159 = arith.mulf %v_157, %v_158 : f64
            %v_160 = arith.addf %v_156, %v_159 : f64
            memref.store %v_160, %v_28[%v_155] : memref<?xf64>
          }
          %v_161 = arith.addi %v_35, %v_137 : index
          %v_162 = arith.addi %v_35, %v_136 : index
          %v_163 = arith.cmpi slt, %v_162, %v_123 : index
          %v_165 = scf.if %v_163 -> (index) {
            %v_164 = memref.load %v_16[%v_162] : memref<?xindex>
            scf.yield %v_164 : index
          } else {
            scf.yield %v_20 : index
          }
          scf.yield %v_161, %v_162, %v_165 : index, index, index
        }
        %v_167 = arith.addi %v_35, %v_128 : index
        %v_168 = arith.maxsi %v_33, %v_167 : index
        scf.for %v_169 = %v_168 to %v_20 step %v_35 {
          %v_170 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_171 = arith.index_cast %v_170 : i64 to index
          %v_172 = arith.muli %v_171, %v_169 : index
          %v_173 = arith.addi %v_116, %v_172 : index
          %v_174 = memref.load %v_28[%v_173] : memref<?xf64>
          memref.store %v_174, %v_28[%v_173] : memref<?xf64>
        }
      }
    }
    %v_175 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_176 = llvm.insertvalue %_A_7, %v_175[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_176, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}
